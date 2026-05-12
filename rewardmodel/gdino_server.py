import os
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import tempfile

app = FastAPI()

# ── 单例 ──────────────────────────────────────────────────────────────────────
_gdino_model = None
_gdino_processor = None
_gdino_device = "cuda"

def _get_gdino():
    global _gdino_model, _gdino_processor, _gdino_device
    if _gdino_model is None:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        model_path = os.environ.get("GDINO_MODEL_PATH", "models/GroundingDINO")
        print(f"[gdino_server] Loading GroundingDINO: {model_path} ...")
        _gdino_processor = AutoProcessor.from_pretrained(model_path)
        _gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
        )
        _gdino_device = "cuda"   # 固定 cpu，需要 gpu 改成 cuda
        _gdino_model = _gdino_model.to(_gdino_device).eval()
        print(f"[gdino_server] GroundingDINO loaded on {_gdino_device}")
    return _gdino_model, _gdino_processor, _gdino_device


@app.post("/detect")
async def detect(
    image: UploadFile = File(...),
    queries: str = Form(...),
    box_threshold: float = Form(0.3),
    text_threshold: float = Form(0.25),
):
    from PIL import Image
    model, processor, device = _get_gdino()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(await image.read())
        tmp_path = tmp.name

    try:
        pil_image = Image.open(tmp_path).convert("RGB")
    except Exception as e:
        os.unlink(tmp_path)
        return JSONResponse({"error": f"Failed to load image: {e}"}, status_code=400)

    query_list = [q.strip() for q in queries.split(",") if q.strip()]
    text_prompt = " . ".join(query_list) + " ."

    with torch.no_grad():
        inputs = processor(images=pil_image, text=text_prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)

        target_sizes = torch.tensor([pil_image.size[::-1]], device=device)

        # 新版 transformers 不支持 box_threshold/text_threshold 参数，手动过滤
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            target_sizes=target_sizes,
        )[0]

    w, h = pil_image.size
    boxes  = results["boxes"].cpu().tolist()
    scores = results["scores"].cpu().tolist()
    labels = results["text_labels"]

    # 手动按阈值过滤
    filtered = [
        (b, s, l)
        for b, s, l in zip(boxes, scores, labels)
        if s >= box_threshold
    ]

    if filtered:
        boxes, scores, labels = zip(*filtered)
        boxes  = list(boxes)
        scores = list(scores)
        labels = list(labels)
    else:
        boxes, scores, labels = [], [], []

    norm_boxes = [[x1/w, y1/h, x2/w, y2/h] for x1, y1, x2, y2 in boxes]

    os.unlink(tmp_path)

    print(f"[gdino_server] Detected {len(norm_boxes)} objects: {labels}")
    return {
        "boxes":  norm_boxes,
        "scores": scores,
        "labels": labels,
    }


@app.get("/health")
def health():
    return {"status": "ok"}