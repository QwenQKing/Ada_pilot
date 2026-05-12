import os
import io
import asyncio
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import uvicorn

app = FastAPI()

# ── 模型全局单例 ──────────────────────────────────────────────────────────────
_model     = None
_processor = None
_device    = None

# ── 动态 Batch 配置 ────────────────────────────────────────────────────────────
_request_queue = None
MAX_BATCH_SIZE = int(os.environ.get("CLIP_MAX_BATCH_SIZE", 16))
BATCH_TIMEOUT  = float(os.environ.get("CLIP_BATCH_TIMEOUT", 0.05))


def load_model():
    global _model, _processor, _device
    model_name = os.environ.get("CLIP_MODEL_PATH", "openai/clip-vit-large-patch14")
    _device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[CLIP Server] Loading {model_name} on {_device} ...")
    _processor = CLIPProcessor.from_pretrained(model_name)
    _model = CLIPModel.from_pretrained(model_name).to(_device).eval()

    if _device == "cuda":
        _model = _model.half()

    print(f"[CLIP Server] Model loaded on {_device}")


# ── 单条推理 ──────────────────────────────────────────────────────────────────
def _single_infer_t(image: Image.Image, caption: str) -> float:
    """CLIP-T：图文相似度"""
    inputs = _processor(
        text=[caption], images=[image],
        return_tensors="pt", padding=True, truncation=True, max_length=77,
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    if _device == "cuda":
        inputs["pixel_values"] = inputs["pixel_values"].half()

    with torch.no_grad():
        outputs = _model(**inputs)
        image_feat = F.normalize(outputs.image_embeds, dim=-1)
        text_feat  = F.normalize(outputs.text_embeds,  dim=-1)
        score = (image_feat * text_feat).sum(dim=-1).item()

    print(f"[CLIP Server] CLIP-T score={score:.6f}")
    return score


def _single_infer_i(image_gen: Image.Image, image_gt: Image.Image) -> float:
    """CLIP-I：图图相似度"""
    inputs = _processor(
        images=[image_gen, image_gt],
        return_tensors="pt", padding=True,
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    if _device == "cuda":
        inputs["pixel_values"] = inputs["pixel_values"].half()

    with torch.no_grad():
        image_feats = F.normalize(_model.get_image_features(**inputs), dim=-1)
        score = (image_feats[0] * image_feats[1]).sum().item()

    print(f"[CLIP Server] CLIP-I score={score:.6f}")
    return score


# ── Batch Worker（对齐 GDINO/HPS）────────────────────────────────────────────
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

        print(f"[CLIP Server] Processing batch of {len(batch)}")

        async def run_one(item):
            fn, args, fut = item
            try:
                result = await loop.run_in_executor(None, fn, *args)
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
    print("[CLIP Server] Batch worker started")


# ── 接口 ──────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": _device,
        "max_batch_size": MAX_BATCH_SIZE,
        "batch_timeout_ms": BATCH_TIMEOUT * 1000,
    }


@app.post("/clip_t")
async def clip_t(
    image: UploadFile = File(...),
    caption: str = Form(...),
):
    """图文相似度"""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        img_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decode failed: {e}")

    loop = asyncio.get_event_loop()
    fut  = loop.create_future()
    await _request_queue.put((_single_infer_t, (pil_image, caption), fut))

    try:
        score = await asyncio.wait_for(fut, timeout=30.0)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Inference timeout")

    normalized = float(max(0.0, min(1.0, (score - 0.15) / 0.25)))
    return {"raw_score": score, "normalized_score": normalized}


@app.post("/clip_i")
async def clip_i(
    image_gen: UploadFile = File(...),
    image_gt:  UploadFile = File(...),
):
    """图图相似度"""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        gen_bytes = await image_gen.read()
        gt_bytes  = await image_gt.read()
        pil_gen = Image.open(io.BytesIO(gen_bytes)).convert("RGB")
        pil_gt  = Image.open(io.BytesIO(gt_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decode failed: {e}")

    loop = asyncio.get_event_loop()
    fut  = loop.create_future()
    await _request_queue.put((_single_infer_i, (pil_gen, pil_gt), fut))

    try:
        score = await asyncio.wait_for(fut, timeout=30.0)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Inference timeout")

    normalized = float(max(0.0, min(1.0, (score - 0.70) / 0.25)))
    return {"raw_score": score, "normalized_score": normalized}


if __name__ == "__main__":
    port = int(os.environ.get("CLIP_SERVER_PORT", 8030))
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)