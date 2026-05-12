import os
import io
import asyncio
import base64
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from PIL import Image
import open_clip
import uvicorn

app = FastAPI()

# ── 模型全局单例 ──────────────────────────────────────────────────────────────
_model      = None
_preprocess = None
_tokenizer  = None
_device     = None

# ── 动态 Batch 配置 ────────────────────────────────────────────────────────────
_request_queue = None
MAX_BATCH_SIZE = int(os.environ.get("HPS_MAX_BATCH_SIZE", 16))
BATCH_TIMEOUT  = float(os.environ.get("HPS_BATCH_TIMEOUT", 0.05))


def load_model():
    global _model, _preprocess, _tokenizer, _device
    model_dir = os.environ.get("HPS_MODEL_DIR", "models/HPSv2")
    hps_weight_path = os.path.join(model_dir, "HPS_v2.1_compressed.pt")

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[HPS Server] Loading model on {_device} ...")

    _model, _, _preprocess = open_clip.create_model_and_transforms(
        model_name="ViT-H-14",
        pretrained=None,
    )
    checkpoint = torch.load(hps_weight_path, map_location="cpu")
    sd = checkpoint.get("state_dict", checkpoint)
    _model.load_state_dict(sd, strict=False)
    _model = _model.to(_device).eval()

    if _device == "cuda":
        _model = _model.half()

    _tokenizer = open_clip.get_tokenizer("ViT-H-14")
    print(f"[HPS Server] Model loaded on {_device}")


# ── 单条推理（供 batch worker 调用）──────────────────────────────────────────
def _single_infer(image_tensor: torch.Tensor, prompt: str) -> float:
    images = image_tensor.unsqueeze(0).to(_device)
    if _device == "cuda":
        images = images.half()

    text_tokens = _tokenizer([prompt]).to(_device)

    with torch.no_grad():
        image_feat = F.normalize(_model.encode_image(images), dim=-1)
        text_feat  = F.normalize(_model.encode_text(text_tokens), dim=-1)
        raw_score  = (image_feat * text_feat).sum().item()

    print(f"[HPS Server] score={raw_score:.6f}")
    return raw_score


# ── Batch Worker（对齐 GDINO，多线程并发）────────────────────────────────────
async def batch_worker():
    loop = asyncio.get_event_loop()

    while True:
        try:
            first = await asyncio.wait_for(_request_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            continue

        batch = [first]

        deadline = loop.time() + BATCH_TIMEOUT
        while len(batch) < MAX_BATCH_SIZE:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                item = await asyncio.wait_for(_request_queue.get(), timeout=remaining)
                batch.append(item)
            except asyncio.TimeoutError:
                break

        print(f"[HPS Server] Processing batch of {len(batch)}")

        # 和 GDINO 完全一致：每条请求在独立线程中并发运行
        async def run_one(item):
            image_tensor, prompt, fut = item
            try:
                result = await loop.run_in_executor(
                    None, _single_infer, image_tensor, prompt
                )
                if not fut.done():
                    fut.set_result(result)
            except Exception as e:
                if not fut.done():
                    fut.set_exception(e)

        await asyncio.gather(*[run_one(item) for item in batch])


# ── 生命周期 ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    global _request_queue
    load_model()
    _request_queue = asyncio.Queue()
    asyncio.create_task(batch_worker())
    print("[HPS Server] Batch worker started")


# ── 接口 ──────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": _device,
        "max_batch_size": MAX_BATCH_SIZE,
        "batch_timeout_ms": BATCH_TIMEOUT * 1000,
    }


@app.post("/score")
async def score(
    image: UploadFile = File(...),
    prompt: str = Form(...),
):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        img_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        image_tensor = _preprocess(pil_image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decode failed: {e}")

    loop = asyncio.get_event_loop()
    fut  = loop.create_future()
    await _request_queue.put((image_tensor, prompt, fut))

    try:
        raw_score = await asyncio.wait_for(fut, timeout=30.0)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Inference timeout")

    normalized = float(max(0.0, min(1.0, (raw_score - 0.20) / 0.15)))
    return {"raw_score": raw_score, "normalized_score": normalized}


if __name__ == "__main__":
    port = int(os.environ.get("HPS_SERVER_PORT", 8020))
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)