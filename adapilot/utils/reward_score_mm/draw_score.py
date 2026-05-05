import re
import os
import torch
import torch.nn.functional as F
import time
from tqdm import tqdm

BASE_WEIGHTS = {
    "clip_t":    0.15,
    "hps":       0.20,
    "aesthetic": 0.20,
    "gdino":     0.25,
    "ocr":       0.25,
}

DATASET_PRIMARY_METRICS = {
    "PosterCraft":      ["clip_t", "hps", "aesthetic", "gdino", "ocr"],
    "lunara-aesthetic":  ["clip_t", "hps", "aesthetic", "gdino"],
    "fine-t2i":         ["clip_t", "hps", "aesthetic", "gdino"],
    "JD-Product":       ["clip_t", "hps", "gdino"],
    "TextAtlas5M":      ["clip_t", "hps", "aesthetic", "ocr"],
    "FinMME":           ["clip_t", "hps", "aesthetic", "ocr"],
    "StoryStream":      ["clip_t", "hps", "aesthetic"],
    "SEED-Story":       ["clip_t", "hps", "aesthetic"],
    "AestheticDataset": ["clip_t", "hps", "aesthetic"],
}

def get_weights(data_source):
    active = DATASET_PRIMARY_METRICS.get(
        data_source, ["clip_t", "hps", "aesthetic", "gdino"]
    )
    raw = {k: BASE_WEIGHTS[k] for k in active if k in BASE_WEIGHTS}
    total = sum(raw.values())
    if total == 0:
        return {k: 1.0 / len(raw) for k in raw}
    return {k: v / total for k, v in raw.items()}

_clip_model = None
_clip_processor = None
_clip_device = None

def _get_clip():
    global _clip_model, _clip_processor, _clip_device
    if _clip_model is None:
        from transformers import CLIPModel, CLIPProcessor
        model_name = os.environ.get("CLIP_MODEL_PATH", "").strip()
        if not model_name or not os.path.isdir(model_name):
            raise RuntimeError(
                f"[draw_score] CLIP_MODEL_PATH not set or not a local directory (got {model_name!r}). "
                f"Use remote: export CLIP_SERVER_URL=http://... "
                f"or local: export CLIP_MODEL_PATH=/path/to/clip-vit-large-patch14"
            )
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE",       "1")
        print(f"[draw_score] Loading CLIP from local: {model_name} ...")
        _clip_processor = CLIPProcessor.from_pretrained(model_name, local_files_only=True)
        _clip_model     = CLIPModel.from_pretrained(model_name, local_files_only=True)
        _clip_device    = "cuda" if torch.cuda.is_available() else "cpu"
        _clip_model     = _clip_model.to(_clip_device).eval()
        print(f"[draw_score] CLIP loaded on {_clip_device}")
    return _clip_model, _clip_processor, _clip_device

def normalize_clip_t(clip_t: float) -> float:
    return float(max(0.0, min(1.0, (clip_t - 0.15) / 0.25)))

def normalize_clip_i(clip_i: float) -> float:
    return float(max(0.0, min(1.0, (clip_i - 0.70) / 0.25)))

def compute_clip_t(image_path: str, caption: str) -> float:
    import requests

    clip_server = os.environ.get("CLIP_SERVER_URL", "")
    max_retries = int(os.environ.get("CLIP_MAX_RETRIES", 3))

    if clip_server:
        for attempt in range(max_retries):
            try:
                with open(image_path, "rb") as f:
                    resp = requests.post(
                        f"{clip_server}/clip_t",
                        files={"image": (os.path.basename(image_path), f, "image/jpeg")},
                        data={"caption": caption},
                        timeout=int(os.environ.get("CLIP_TIMEOUT", 30)),
                        proxies={"http": None, "https": None},
                    )
                resp.raise_for_status()
                data = resp.json()
                print(f"[draw_score] CLIP-T remote raw={data['raw_score']:.6f} normalized={data['normalized_score']:.4f}")
                return data["normalized_score"]
            except requests.exceptions.Timeout:
                wait = 5 * (attempt + 1)
                print(f"[draw_score] CLIP-T timeout, retry {attempt+1}/{max_retries} after {wait}s")
                if attempt < max_retries - 1:
                    time.sleep(wait)
            except Exception as e:
                wait = 5 * (attempt + 1)
                print(f"[draw_score] CLIP-T remote failed ({e}), retry {attempt+1}/{max_retries} after {wait}s")
                if attempt < max_retries - 1:
                    time.sleep(wait)

        print("[draw_score] CLIP-T all retries failed, returning 0.0 (no local fallback when server configured)")
        return 0.0

    from PIL import Image
    model, processor, device = _get_clip()
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[draw_score] Failed to load image {image_path}: {e}")
        return 0.0

    with torch.no_grad():
        inputs = processor(
            text=[caption], images=[image],
            return_tensors="pt", padding=True, truncation=True, max_length=77,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        image_feat = F.normalize(outputs.image_embeds, dim=-1)
        text_feat  = F.normalize(outputs.text_embeds,  dim=-1)
        clip_t = (image_feat * text_feat).sum(dim=-1).item()

    return normalize_clip_t(clip_t)

def compute_clip_i(image_path_gen: str, image_path_gt: str) -> float:
    import requests

    clip_server = os.environ.get("CLIP_SERVER_URL", "")
    max_retries = int(os.environ.get("CLIP_MAX_RETRIES", 3))

    if clip_server:
        for attempt in range(max_retries):
            try:
                with open(image_path_gen, "rb") as f_gen, open(image_path_gt, "rb") as f_gt:
                    resp = requests.post(
                        f"{clip_server}/clip_i",
                        files={
                            "image_gen": (os.path.basename(image_path_gen), f_gen, "image/jpeg"),
                            "image_gt":  (os.path.basename(image_path_gt),  f_gt,  "image/jpeg"),
                        },
                        timeout=int(os.environ.get("CLIP_TIMEOUT", 30)),
                        proxies={"http": None, "https": None},
                    )
                resp.raise_for_status()
                data = resp.json()
                print(f"[draw_score] CLIP-I remote raw={data['raw_score']:.6f} normalized={data['normalized_score']:.4f}")
                return data["normalized_score"]
            except requests.exceptions.Timeout:
                wait = 5 * (attempt + 1)
                print(f"[draw_score] CLIP-I timeout, retry {attempt+1}/{max_retries} after {wait}s")
                if attempt < max_retries - 1:
                    time.sleep(wait)
            except Exception as e:
                wait = 5 * (attempt + 1)
                print(f"[draw_score] CLIP-I remote failed ({e}), retry {attempt+1}/{max_retries} after {wait}s")
                if attempt < max_retries - 1:
                    time.sleep(wait)

        print("[draw_score] CLIP-I all retries failed, returning 0.0 (no local fallback when server configured)")
        return 0.0

    from PIL import Image
    model, processor, device = _get_clip()
    try:
        image_gen = Image.open(image_path_gen).convert("RGB")
        image_gt  = Image.open(image_path_gt).convert("RGB")
    except Exception as e:
        print(f"[draw_score] Failed to load images for CLIP-I: {e}")
        return 0.0

    with torch.no_grad():
        inputs = processor(images=[image_gen, image_gt], return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        image_feats = F.normalize(model.get_image_features(**inputs), dim=-1)
        clip_i = (image_feats[0] * image_feats[1]).sum().item()

    return normalize_clip_i(clip_i)

_hps_model      = None
_hps_preprocess = None
_hps_tokenizer  = None
_hps_device     = None

def _get_hps():
    global _hps_model, _hps_preprocess, _hps_tokenizer, _hps_device
    if _hps_model is None:
        import open_clip

        model_dir = os.environ.get("HPS_MODEL_DIR", "").strip()
        if not model_dir or not os.path.isdir(model_dir):
            raise RuntimeError(
                f"[draw_score] HPS_MODEL_DIR not set or not a local directory (got {model_dir!r}). "
                f"Use remote: export HPS_SERVER_URL=http://... "
                f"or local: export HPS_MODEL_DIR=/path/to/HPSv2"
            )
        hps_weight_path = os.path.join(model_dir, "HPS_v2.1_compressed.pt")
        if not os.path.isfile(hps_weight_path):
            raise RuntimeError(
                f"[draw_score] HPS weight file not found: {hps_weight_path}"
            )
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE",       "1")

        _hps_device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[draw_score] Loading HPS v2.1 from local: {model_dir} ...")

        _hps_model, _, _hps_preprocess = open_clip.create_model_and_transforms(
            model_name="ViT-H-14",
            pretrained=None,
        )

        checkpoint = torch.load(hps_weight_path, map_location="cpu")
        sd = checkpoint.get("state_dict", checkpoint)
        _hps_model.load_state_dict(sd, strict=False)
        _hps_model = _hps_model.to(_hps_device).eval()

        _hps_tokenizer = open_clip.get_tokenizer("ViT-H-14")

        with torch.no_grad():
            dummy_img = torch.zeros(1, 3, 224, 224).to(_hps_device)
            dummy_txt = _hps_tokenizer(["test"]).to(_hps_device)
            i_feat = F.normalize(_hps_model.encode_image(dummy_img), dim=-1)
            t_feat = F.normalize(_hps_model.encode_text(dummy_txt),  dim=-1)
            score  = (i_feat * t_feat).sum().item()
            print(f"[HPS] Sanity check score: {score:.6f}")

        print(f"[draw_score] HPS v2.1 loaded on {_hps_device}")
    return _hps_model, _hps_preprocess, _hps_tokenizer, _hps_device

def compute_hps_score(image_path: str, prompt: str) -> float:
    import requests

    hps_server = os.environ.get("HPS_SERVER_URL", "")
    max_retries = int(os.environ.get("HPS_MAX_RETRIES", 3))

    if hps_server:
        for attempt in range(max_retries):
            try:
                with open(image_path, "rb") as f:
                    resp = requests.post(
                        f"{hps_server}/score",
                        files={"image": (os.path.basename(image_path), f, "image/jpeg")},
                        data={"prompt": prompt},
                        timeout=int(os.environ.get("HPS_TIMEOUT", 30)),
                        proxies={"http": None, "https": None},
                    )
                resp.raise_for_status()
                data = resp.json()
                score = data["normalized_score"]
                print(f"[draw_score] HPS remote raw={data['raw_score']:.6f} normalized={score:.4f}")
                return score
            except requests.exceptions.Timeout:
                wait = 5 * (attempt + 1)
                print(f"[draw_score] HPS timeout, retry {attempt+1}/{max_retries} after {wait}s")
                if attempt < max_retries - 1:
                    time.sleep(wait)
            except Exception as e:
                wait = 5 * (attempt + 1)
                print(f"[draw_score] HPS remote failed ({e}), retry {attempt+1}/{max_retries} after {wait}s")
                if attempt < max_retries - 1:
                    time.sleep(wait)

        print("[draw_score] HPS remote all retries failed, returning 0.0 (no local fallback when server configured)")
        return 0.0

    from PIL import Image
    model, preprocess, tokenizer, device = _get_hps()

    try:
        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    except Exception as e:
        print(f"[draw_score] Failed to load image for HPS: {e}")
        return 0.0

    text = tokenizer([prompt]).to(device)

    with torch.no_grad():
        image_feat = F.normalize(model.encode_image(image), dim=-1)
        text_feat  = F.normalize(model.encode_text(text),   dim=-1)
        score = (image_feat * text_feat).sum().item()

    print(f"[draw_score] HPS local raw score={score:.6f}")
    normalized = (score - 0.20) / 0.15
    return float(max(0.0, min(1.0, normalized)))

def _normalize_text(text: str) -> str:
    import re
    text = text.lower()
    text = re.sub(r'[.,;:!?\'"\-()\[\]{}`…]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def _compute_text_metrics(gt_text: str, ocr_text: str) -> dict:
    gt  = _normalize_text(gt_text)
    ocr = _normalize_text(ocr_text)

    if not gt and not ocr:
        return {"f_score": 1.0, "precision": 1.0, "recall": 1.0, "accuracy": 1.0, "C": 0, "I": 0, "D": 0, "S": 0}
    if not gt:
        return {"f_score": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0, "C": 0, "I": len(ocr), "D": 0, "S": 0}
    if not ocr:
        return {"f_score": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0, "C": 0, "I": 0, "D": len(gt), "S": 0}

    n, m = len(gt), len(ocr)

    dp = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = (0, 0, 0, 0, 0)
    for i in range(1, n + 1):
        dp[i][0] = (i, 0, 0, i, 0)
    for j in range(1, m + 1):
        dp[0][j] = (j, 0, j, 0, 0)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c, ii, d, s = dp[i-1][j-1][1:]
            if gt[i-1] == ocr[j-1]:
                match  = (dp[i-1][j-1][0],     c+1, ii,   d,   s)
            else:
                match  = (dp[i-1][j-1][0] + 1, c,   ii,   d,   s+1)
            c2, ii2, d2, s2 = dp[i-1][j][1:]
            delete = (dp[i-1][j][0] + 1,        c2,  ii2,  d2+1, s2)
            c3, ii3, d3, s3 = dp[i][j-1][1:]
            insert = (dp[i][j-1][0] + 1,        c3,  ii3+1, d3,  s3)
            dp[i][j] = min(match, delete, insert, key=lambda x: x[0])

    _, C, I, D, S = dp[n][m]
    T = C + I + D + S
    precision = C / (C + I + S) if (C + I + S) > 0 else 0.0
    recall    = C / (C + D + S) if (C + D + S) > 0 else 0.0
    accuracy  = C / T           if T > 0           else 0.0
    f_score   = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "C": C, "I": I, "D": D, "S": S,
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "accuracy":  round(accuracy,  4),
        "f_score":   round(f_score,   4),
    }

def glm_ocr_judge(generated_image_path: str, caption: str, data_source: str = "") -> float:
    import base64
    import requests

    glm_api_base = os.environ.get("GLM_OCR_API_BASE", "")
    glm_model    = os.environ.get("GLM_OCR_MODEL", "glm-ocr")
    txt_api_base = os.environ.get("VLM_API_BASE", "")
    txt_model    = os.environ.get("VLM_MODEL", "")

    gt_text = ""
    if data_source == "ShowTable":
        gt_text = _normalize_text(caption)
        print(f"[glm_ocr_judge] ShowTable: using caption directly as GT: '{gt_text[:80]}'")
        if not gt_text:
            print("[glm_ocr_judge] ShowTable: empty caption → OCR score = 1.0")
            return 1.0
    else:
        try:
            gt_prompt = (
                f"You are an extraction engine.\n"
                f"Task: extract ALL text strings that should visibly appear in the image, "
                f"including: titles, subtitles, slogans, dates, venue, author names, labels, "
                f"captions, button text, and any other readable text element.\n"
                f"IMPORTANT: Only extract text that is explicitly quoted or described as visible rendered text "
                f"in the image (e.g. 'The title \"NORMAL PEOPLE\" is displayed at the top'). "
                f"If the prompt is a pure scene/object description with no visible text specified, "
                f"you MUST output: <text>None</text>\n"
                f"Rules:\n"
                f"- Extract text ONLY from the following prompt, do NOT infer or fabricate anything.\n"
                f"- Preserve the original spelling and casing exactly as written in the prompt.\n"
                f"- Order extracted strings top→bottom, left→right as they would appear spatially.\n"
                f"- Concatenate all strings into a single space-separated string.\n"
                f"- Output ONLY in this format, no explanation: <text>...</text>\n"
                f"Prompt: {caption}"
            )
            resp = requests.post(
                f"{txt_api_base}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": txt_model,
                    "messages": [{"role": "user", "content": gt_prompt}],
                    "max_tokens": 256,
                    "chat_template_kwargs": {"thinking": False},
                },
                timeout=30,
                proxies={"http": None, "https": None},
            )
            resp.raise_for_status()
            gt_text = resp.json()["choices"][0]["message"]["content"].strip()
            gt_text = gt_text.replace("<|user|>", "").replace("<|assistant|>", "").strip()
            match = re.search(r'<text>(.*?)</text>', gt_text, re.DOTALL)
            if match:
                gt_text = match.group(1).strip()
                print(f"[glm_ocr_judge] GT text extracted: '{gt_text[:80]}'")
            else:
                print(f"[glm_ocr_judge] No <text> tag found in GT response: {gt_text[:100]}")
                return 0.0

        except Exception as e:
            print(f"[glm_ocr_judge] GT extraction failed: {e}")
            return 0.0

        if not gt_text or gt_text.lower() == "none":
            print("[glm_ocr_judge] No text expected in caption → OCR score = 1.0")
            return 1.0

    try:
        with open(generated_image_path, "rb") as f:
            gen_b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"[glm_ocr_judge] Failed to encode image: {e}")
        return 0.0

    ocr_text = ""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{glm_api_base}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": glm_model,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{gen_b64}"}},
                            {"type": "text", "text": "Text Recognition:"},
                        ],
                    }],
                    "max_tokens": 8192,
                },
                timeout=120,
                proxies={"http": None, "https": None},
            )
            if resp.status_code == 503:
                wait = 10 * (attempt + 1)
                print(f"[glm_ocr_judge] Server busy, retry {attempt+1}/{max_retries} after {wait}s")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            ocr_text = resp.json()["choices"][0]["message"]["content"].strip()
            ocr_text = ocr_text.replace("<|user|>", "").replace("<|assistant|>", "").strip()
            print(f"[glm_ocr_judge] OCR text recognized: '{ocr_text[:80]}'")
            break
        except requests.exceptions.Timeout:
            wait = 10 * (attempt + 1)
            print(f"[glm_ocr_judge] OCR timeout, retry {attempt+1}/{max_retries} after {wait}s")
            if attempt < max_retries - 1:
                time.sleep(wait)
        except Exception as e:
            print(f"[glm_ocr_judge] OCR request failed: {e}")
            return 0.0

    if not ocr_text:
        print("[glm_ocr_judge] OCR text is empty")
        return 0.0

    metrics = _compute_text_metrics(gt_text, ocr_text)
    f_score = metrics["f_score"]
    print(
        f"[glm_ocr_judge] GT='{gt_text[:60]}' | OCR='{ocr_text[:60]}' | "
        f"C={metrics['C']} I={metrics['I']} D={metrics['D']} S={metrics['S']} | "
        f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} "
        f"f_score={f_score:.4f}"
    )
    return f_score

def llm_aesthetic_judge(image_path: str) -> float:
    import base64
    import json
    import requests

    api_base = os.environ.get("VLM_API_BASE", "")
    model    = os.environ.get("VLM_MODEL", "")
    max_retries = int(os.environ.get("LLM_AESTHETIC_MAX_RETRIES", 3))

    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"[llm_aesthetic_judge] Failed to encode image: {e}")
        return 0.0

    prompt = """You are an extremely discerning visual-element judge. Scrutinize every figure, chart, and image for any visual, stylistic, or layout issue. Evaluate not only resolution, clarity, and style consistency, but also the overall arrangement of visual components (text blocks, headings, figures, whitespace, alignment) that affect readability and comprehension. Always look for even subtle flaws: low contrast, imperfect resolutions, slightly inconsistent styles, crowded or mislabeled legends, uneven spacing, or misaligned elements. Be wary of awarding high scores unless the visuals truly meet the strictest standards.
Return ONLY the final JSON object based on these internal instructions:
Five-Point Scale
1 Point:
- Graphics are blurry, pixelated, or illegible.
- Color choices severely hinder interpretation.
- Visuals may significantly detract from comprehension.
- Layout is highly disorganized; elements overlap, margins are violated, or reading path is almost impossible.
2 Points:
- At least one graphic is clear, while others suffer from poor resolution or style.
- Legends or labels are missing or too small to read comfortably.
- Color schemes create some confusion or difficulty.
- Layout shows some structure but suffers from inconsistent alignment, overcrowding, or uneven spacing that interrupts reading flow.
3 Points:
- Most graphics are legible and relevant, but have notable issues with consistency, sizing, or clarity.
- Some mismatches in style or color usage detract from cohesion.
- Minor but noticeable labeling/legend shortcomings.
- Layout has recognizable structure with mostly consistent alignment and spacing; minor distractions remain (slightly cramped text, small alignment slips).
4 Points:
- High-quality graphics with generally consistent styling.
- Clear legends and color schemes aid interpretation.
- Any remaining flaws are slight and do not significantly hinder understanding.
- Layout is well-organized with logical reading flow; font sizes, spacing, and alignment are mostly appropriate; subtle imperfections may exist.
5 Points:
- Rarely awarded; strictly reserved for publication-grade visuals.
- Crisp resolution with no instances of blurriness.
- Harmonious color palette, impeccable labeling, and an exceptionally consistent style.
- Layout is pristine and professional; seamless alignment, balanced spacing, and a guided reading path maximize readability and comprehension.
YOU MUST output ONLY the following JSON, nothing else, no markdown, no explanation, no step labels:
{"reason": "...", "score": int}
"""

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": 512,
        "n": 1,
        "chat_template_kwargs": {"thinking": False},
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{api_base}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=60,
            )
            if resp.status_code == 503:
                wait = 10 * (attempt + 1)
                print(f"[llm_aesthetic_judge] Server busy, retry {attempt+1}/{max_retries} after {wait}s")
                if attempt < max_retries - 1:
                    time.sleep(wait)
                continue
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()

            json_start = content.find('{')
            json_end   = content.rfind('}') + 1
            if json_start == -1 or json_end == 0:
                print(f"[llm_aesthetic_judge] No JSON found: {content[:200]}")
                return 0.0

            result = json.loads(content[json_start:json_end])
            raw_score = float(result.get("score", 0))
            reason    = result.get("reason", "")

            normalized = (raw_score - 1.0) / 4.0
            normalized = float(max(0.0, min(1.0, normalized)))

            print(
                f"[llm_aesthetic_judge] raw_score={raw_score:.1f} "
                f"normalized={normalized:.4f} | reason={reason[:80]}"
            )
            return normalized

        except requests.exceptions.Timeout:
            wait = 10 * (attempt + 1)
            print(f"[llm_aesthetic_judge] Timeout, retry {attempt+1}/{max_retries} after {wait}s")
            if attempt < max_retries - 1:
                time.sleep(wait)
        except Exception as e:
            wait = 5 * (attempt + 1)
            print(f"[llm_aesthetic_judge] Failed ({e}), retry {attempt+1}/{max_retries} after {wait}s")
            if attempt < max_retries - 1:
                time.sleep(wait)

    print("[llm_aesthetic_judge] All retries failed, returning 0.0")
    return 0.0

def extract_entities_with_llm(caption: str) -> list:
    import requests, json

    api_base = os.environ.get("VLM_API_BASE", "")
    model    = os.environ.get("VLM_MODEL", "")
    max_retries = int(os.environ.get("VLM_ENTITY_MAX_RETRIES", 3))

    prompt = f"""You are an entity extractor.
Given a text prompt, extract structured information.
Return a valid JSON array. Each item must have the following format:
{{
  "label": "short generic detectable category name only, avoid descriptive adjectives",
  "count": "the number of instances of this entity",
  "spatial_relation": ["spatial relations involving this entity"]
}}

Rules:
- Extract only visually grounded entities that can be localized with a bounding box
- Only extract high-level physical objects or animals that a general object detector can reliably detect, such as person, car, dog, chair, building, bottle
- Do NOT extract text, logo, letters, words, symbols, signage, poster titles, brand marks, graphic design elements, or other printed/overlay visual elements
- Do NOT extract abstract regions or non-object areas such as background, sky, lighting, shadow, texture, atmosphere, composition, or layout anchors
- Do NOT extract fine-grained body parts, facial details, clothing details, accessories that are too small or ambiguous, or abstract elements
- Separate entity names from descriptive attributes
- Only keep spatial relations that can be verified from bounding boxes, such as left of, right of, above, below, inside, overlap, near, in front of, behind
- If an element is unlikely to be reliably detected by GroundingDINO as a standalone object, do not include it
- Output concise and valid JSON only
- Do not include explanations or extra text
Prompt:
"{caption}"
"""

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{api_base}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 512,
                    "chat_template_kwargs": {"thinking": False},
                },
                timeout=30,
            )
            if resp.status_code == 503:
                wait = 5 * (attempt + 1)
                print(f"[extract_entities] Server busy, retry {attempt+1}/{max_retries} after {wait}s")
                if attempt < max_retries - 1:
                    time.sleep(wait)
                continue
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()

            start = content.find('[')
            end   = content.rfind(']') + 1
            if start == -1 or end == 0:
                print(f"[extract_entities] No JSON array found in response: {content[:200]}")
                return []

            raw_list = json.loads(content[start:end])

            entities = []
            for item in raw_list:
                if not isinstance(item, dict) or "label" not in item:
                    continue
                try:
                    count = int(item.get("count", 1))
                    if count <= 0:
                        count = 1
                except (ValueError, TypeError):
                    count = 1
                spatial = item.get("spatial_relation", [])
                if isinstance(spatial, str):
                    spatial = [spatial]
                label = str(item["label"]).strip().lower()
                label = re.sub(r'["\'].*?["\']', '', label).strip()
                label = re.sub(r'\s+', ' ', label).strip()
                if not label:
                    continue
                entities.append({
                    "label":            label,
                    "count":            count,
                    "spatial_relation": [s.strip() for s in spatial if s.strip()],
                })

            print(f"[extract_entities] Extracted {len(entities)} entities: {entities}")
            return entities

        except requests.exceptions.Timeout:
            wait = 5 * (attempt + 1)
            print(f"[extract_entities] Timeout, retry {attempt+1}/{max_retries} after {wait}s")
            if attempt < max_retries - 1:
                time.sleep(wait)
        except Exception as e:
            wait = 5 * (attempt + 1)
            print(f"[extract_entities] Failed ({e}), retry {attempt+1}/{max_retries} after {wait}s")
            if attempt < max_retries - 1:
                time.sleep(wait)

    print("[extract_entities] All retries failed, returning empty list")
    return []

_gdino_model = None
_gdino_processor = None
_gdino_device = None

def _get_gdino():
    global _gdino_model, _gdino_processor, _gdino_device
    if _gdino_model is None:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        model_path = os.environ.get("GDINO_MODEL_PATH", "models/GroundingDINO")
        print(f"[draw_score] Loading GroundingDINO locally: {model_path} ...")
        _gdino_processor = AutoProcessor.from_pretrained(model_path)
        _gdino_device = "cuda" if torch.cuda.is_available() else "cpu"
        _gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map=_gdino_device,
        )
        _gdino_model = _gdino_model.eval()
        print(f"[draw_score] GroundingDINO loaded on {_gdino_device}")
    return _gdino_model, _gdino_processor, _gdino_device

def detect_objects(image_path: str, queries: list,
                   box_threshold: float = 0.3,
                   text_threshold: float = 0.25) -> dict:
    import requests as req

    if not queries:
        return {"boxes": [], "scores": [], "labels": [], "raw_queries": []}

    gdino_server = os.environ.get("GDINO_SERVER_URL", "")
    max_retries = int(os.environ.get("GDINO_MAX_RETRIES", 3))

    if gdino_server:
        for attempt in range(max_retries):
            try:
                with open(image_path, "rb") as f:
                    resp = req.post(
                        f"{gdino_server}/detect",
                        files={"image": (os.path.basename(image_path), f, "image/jpeg")},
                        data={
                            "queries":        ",".join(queries),
                            "box_threshold":  box_threshold,
                            "text_threshold": text_threshold,
                        },
                        timeout=int(os.environ.get("GDINO_TIMEOUT", 60)),
                    )
                resp.raise_for_status()
                result = resp.json()
                result["raw_queries"] = queries
                print(f"[draw_score] GDINO remote detected {len(result.get('labels', []))} objects: {result.get('labels', [])}")
                return result
            except req.exceptions.Timeout:
                wait = 10 * (attempt + 1)
                print(f"[draw_score] GDINO timeout, retry {attempt+1}/{max_retries} after {wait}s")
                if attempt < max_retries - 1:
                    time.sleep(wait)
            except Exception as e:
                wait = 5 * (attempt + 1)
                print(f"[draw_score] GDINO remote failed ({e}), retry {attempt+1}/{max_retries} after {wait}s")
                if attempt < max_retries - 1:
                    time.sleep(wait)

        print("[draw_score] GDINO all retries failed, returning empty detection (no local fallback when server configured)")
        return {"boxes": [], "scores": [], "labels": [], "raw_queries": queries}

    from PIL import Image
    model, processor, device = _get_gdino()

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[draw_score] GroundingDINO: failed to load image {image_path}: {e}")
        return {"boxes": [], "scores": [], "labels": [], "raw_queries": queries}

    text_prompt = " . ".join(queries) + " ."

    try:
        with torch.no_grad():
            inputs = processor(images=image, text=text_prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)

            target_sizes = torch.tensor([image.size[::-1]], device=device)
            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=target_sizes,
            )[0]
    except RuntimeError as e:
        print(f"[draw_score] GDINO local inference failed: {e}, returning empty detection")
        return {"boxes": [], "scores": [], "labels": [], "raw_queries": queries}

    boxes  = results["boxes"].cpu().tolist()
    scores = results["scores"].cpu().tolist()
    labels = results["text_labels"]

    w, h = image.size
    norm_boxes = [[x1/w, y1/h, x2/w, y2/h] for x1, y1, x2, y2 in boxes]

    print(f"[draw_score] GDINO local detected {len(boxes)} objects: {labels}")
    return {
        "boxes":       norm_boxes,
        "scores":      scores,
        "labels":      labels,
        "raw_queries": queries,
    }

def score_object_existence(detection: dict, entities: list) -> float:
    if not entities:
        return 1.0

    detected_labels = [l.lower() for l in detection["labels"]]

    hit = 0
    for e in entities:
        q = e["label"].lower()
        matched = any(
            q == dl or q in dl or dl in q
            for dl in detected_labels
        )
        if matched:
            hit += 1
        else:
            print(f"[draw_score] Existence MISS: '{q}' not found in {detected_labels}")

    score = hit / len(entities)
    print(f"[draw_score] Existence: {hit}/{len(entities)} entities hit → {score:.3f}")
    return score

def _score_count_from_entities(detection: dict, entities: list) -> float:
    if not entities:
        return 1.0

    labels = [l.lower() for l in detection["labels"]]
    errors = []
    for e in entities:
        expected = e["count"]
        obj      = e["label"]
        detected = sum(1 for l in labels if obj in l or l in obj)
        if detected == 0:
            errors.append(1.0)
        elif expected == 0:
            errors.append(0.0)
        else:
            errors.append(min(1.0, abs(detected - expected) / expected))

    score = 1.0 - sum(errors) / len(errors)
    print(f"[draw_score] Count (entities): score={score:.3f}")
    return score

def _score_spatial_from_entities(detection: dict, entities: list) -> float:
    boxes  = detection["boxes"]
    labels = [l.lower() for l in detection["labels"]]

    if not boxes:
        return 1.0 if not any(e["spatial_relation"] for e in entities) else 0.0

    RELATION_MAP = {
        "left of":    lambda a, b: a[0] < b[0],
        "right of":   lambda a, b: a[0] > b[0],
        "above":      lambda a, b: a[1] < b[1],
        "below":      lambda a, b: a[1] > b[1],
        "in front of": lambda a, b: True,
        "near":        lambda a, b: abs(a[0]-b[0]) < 0.3 and abs(a[1]-b[1]) < 0.3,
        "inside":      lambda a, b: True,
        "overlap":     lambda a, b: True,
    }

    def get_center(name):
        for i, label in enumerate(labels):
            if name in label or label in name:
                x1, y1, x2, y2 = boxes[i]
                return ((x1+x2)/2, (y1+y2)/2)
        return None

    triples = []
    for e in entities:
        for rel_str in e["spatial_relation"]:
            rel_str = rel_str.lower().strip()
            for rel_key in RELATION_MAP:
                if rel_key in rel_str:
                    parts = rel_str.split(rel_key, 1)
                    obj_a = parts[0].strip() or e["label"]
                    obj_b = parts[1].strip()
                    if obj_b:
                        triples.append((obj_a, rel_key, obj_b))
                    break

    if not triples:
        return 1.0

    correct = 0
    for obj_a, rel, obj_b in triples:
        ca = get_center(obj_a)
        cb = get_center(obj_b)
        if ca is None or cb is None:
            continue
        if RELATION_MAP[rel](ca, cb):
            correct += 1

    score = correct / len(triples)
    print(f"[draw_score] Spatial (entities): {correct}/{len(triples)} → {score:.3f}")
    return score

def compute_gdino_score(image_path: str, caption: str,
                        existence_weight: float = 0.5,
                        spatial_weight:   float = 0.25,
                        count_weight:     float = 0.25) -> float:

    entities  = extract_entities_with_llm(caption)
    queries   = [e["label"] for e in entities]
    detection = detect_objects(image_path, queries)

    existence = score_object_existence(detection, entities)
    count   = _score_count_from_entities(detection, entities)
    spatial = _score_spatial_from_entities(detection, entities)

    score = (
        existence_weight * existence
      + spatial_weight   * spatial
      + count_weight     * count
    )
    print(
        f"[draw_score] GDINO | existence={existence:.3f} spatial={spatial:.3f} "
        f"count={count:.3f} → gdino_score={score:.3f}"
    )
    return score

def compute_clip_t_batch(image_paths: list, captions: list,
                          max_workers: int = None) -> list:
    if os.environ.get("CLIP_SERVER_URL", ""):
        from concurrent.futures import ThreadPoolExecutor, as_completed
        if max_workers is None:
            max_workers = int(os.environ.get("CLIP_MAX_WORKERS", 8))

        results = [0.0] * len(image_paths)
        valid_idx = [i for i, p in enumerate(image_paths) if p and os.path.exists(str(p))]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(compute_clip_t, image_paths[i], captions[i]): i
                for i in valid_idx
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"[draw_score] CLIP-T batch remote failed idx={idx}: {e}")
        return results

    from PIL import Image
    model, processor, device = _get_clip()

    images, valid_idx = [], []
    for i, p in enumerate(image_paths):
        try:
            images.append(Image.open(p).convert("RGB"))
            valid_idx.append(i)
        except Exception as e:
            print(f"[draw_score] Failed to load image {p}: {e}")

    results = [0.0] * len(image_paths)
    if not images:
        return results

    valid_captions = [captions[i] for i in valid_idx]
    try:
        with torch.no_grad():
            inputs = processor(
                text=valid_captions, images=images,
                return_tensors="pt", padding=True, truncation=True, max_length=77,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            image_feats = F.normalize(outputs.image_embeds, dim=-1)
            text_feats  = F.normalize(outputs.text_embeds,  dim=-1)
            scores = (image_feats * text_feats).sum(dim=-1).cpu().tolist()
            if isinstance(scores, float):
                scores = [scores]
        for i, idx in enumerate(valid_idx):
            results[idx] = normalize_clip_t(scores[i])
    except Exception as e:
        import traceback
        print(f"[draw_score] CLIP-T batch local inference failed: {e}")
        traceback.print_exc()

    return results

def compute_hps_score_batch(image_paths: list, prompts: list,
                             max_workers: int = None) -> list:
    if os.environ.get("HPS_SERVER_URL", ""):
        from concurrent.futures import ThreadPoolExecutor, as_completed
        if max_workers is None:
            max_workers = int(os.environ.get("HPS_MAX_WORKERS", 8))

        results = [0.0] * len(image_paths)
        valid_idx = [i for i, p in enumerate(image_paths) if p and os.path.exists(str(p))]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(compute_hps_score, image_paths[i], prompts[i]): i
                for i in valid_idx
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"[draw_score] HPS batch remote failed idx={idx}: {e}")
        return results

    from PIL import Image
    model, preprocess, tokenizer, device = _get_hps()

    results = [0.0] * len(image_paths)
    images, valid_idx = [], []

    for i, p in enumerate(image_paths):
        try:
            images.append(preprocess(Image.open(p).convert("RGB")))
            valid_idx.append(i)
        except Exception as e:
            print(f"[draw_score] Failed to load image for HPS idx={i}: {e}")

    if not images:
        return results

    valid_prompts = [prompts[i] for i in valid_idx]
    image_tensor = torch.stack(images).to(device)
    text_tokens  = tokenizer(valid_prompts).to(device)

    try:
        with torch.no_grad():
            image_feats = F.normalize(model.encode_image(image_tensor), dim=-1)
            text_feats  = F.normalize(model.encode_text(text_tokens),   dim=-1)
            scores = (image_feats * text_feats).sum(dim=-1).cpu().tolist()
            if isinstance(scores, float):
                scores = [scores]
        for i, idx in enumerate(valid_idx):
            raw = scores[i]
            results[idx] = float(max(0.0, min(1.0, (raw - 0.20) / 0.15)))
    except Exception as e:
        import traceback
        print(f"[draw_score] HPS batch local inference failed: {e}")
        traceback.print_exc()

    return results

def compute_llm_ocr_batch(image_paths: list, captions: list,
                           data_sources: list = None,
                           max_workers: int = None) -> list:
    if max_workers is None:
        max_workers = int(os.environ.get("LLM_OCR_MAX_WORKERS", 8))
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if data_sources is None:
        data_sources = [""] * len(image_paths)

    results = [0.0] * len(image_paths)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(glm_ocr_judge, p, c, ds): i
            for i, (p, c, ds) in enumerate(zip(image_paths, captions, data_sources))
            if p and os.path.exists(str(p)) and c
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                results[idx] = result if result is not None else 0.0
            except Exception as e:
                print(f"[draw_score] OCR batch failed idx={idx}: {e}")
                results[idx] = 0.0
    return results

def compute_llm_aesthetic_batch(image_paths: list,
                                 max_workers: int = None) -> list:
    if max_workers is None:
        max_workers = int(os.environ.get("LLM_AESTHETIC_MAX_WORKERS", 8))
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = [0.0] * len(image_paths)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(llm_aesthetic_judge, p): i
            for i, p in enumerate(image_paths)
            if p and os.path.exists(str(p))
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"[llm_aesthetic_judge] Batch failed idx={idx}: {e}")
    return results

def compute_gdino_batch(image_paths: list, captions: list,
                         max_workers: int = None) -> list:
    if max_workers is None:
        max_workers = int(os.environ.get("GDINO_MAX_WORKERS", 4))
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = [0.0] * len(image_paths)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(compute_gdino_score, p, c): i
            for i, (p, c) in enumerate(zip(image_paths, captions))
            if p and os.path.exists(str(p)) and c
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = float(future.result())
            except Exception as e:
                print(f"[draw_score] GDINO batch failed idx={idx}: {e}")
                results[idx] = 0.0
    return results

def _check_round(assistant_block: str, kind: str) -> bool:
    r = assistant_block.strip()

    if kind == 'first':
        ok_count = (
            r.count('<reasoning>') == 1 and r.count('</reasoning>') == 1 and
            r.count('<paint>')     == 1 and r.count('</paint>')     == 1 and
            r.count('<assess>')    == 0 and r.count('</assess>')    == 0 and
            r.count('<finish>')    == 0 and r.count('</finish>')    == 0
        )
        if not ok_count:
            return False
        reasoning = re.search(r'<reasoning>(.*?)</reasoning>', r, re.DOTALL).group(1).strip()
        paint     = re.search(r'<paint>(.*?)</paint>',         r, re.DOTALL).group(1).strip()
        return bool(reasoning) and bool(paint)

    if kind == 'middle':
        ok_count = (
            r.count('<reasoning>') == 1 and r.count('</reasoning>') == 1 and
            r.count('<assess>')    == 1 and r.count('</assess>')    == 1 and
            r.count('<paint>')     == 1 and r.count('</paint>')     == 1 and
            r.count('<finish>')    == 0 and r.count('</finish>')    == 0
        )
        if not ok_count:
            return False
        reasoning = re.search(r'<reasoning>(.*?)</reasoning>', r, re.DOTALL).group(1).strip()
        assess    = re.search(r'<assess>(.*?)</assess>',       r, re.DOTALL).group(1).strip()
        paint     = re.search(r'<paint>(.*?)</paint>',         r, re.DOTALL).group(1).strip()
        return bool(reasoning) and bool(assess) and bool(paint)

    if kind == 'last':
        ok_count = (
            r.count('<reasoning>') == 1 and r.count('</reasoning>') == 1 and
            r.count('<assess>')    == 1 and r.count('</assess>')    == 1 and
            r.count('<finish>')    == 1 and r.count('</finish>')    == 1 and
            r.count('<paint>')     == 0 and r.count('</paint>')     == 0
        )
        if not ok_count:
            return False
        reasoning = re.search(r'<reasoning>(.*?)</reasoning>', r, re.DOTALL).group(1).strip()
        assess    = re.search(r'<assess>(.*?)</assess>',       r, re.DOTALL).group(1).strip()
        finish    = re.search(r'<finish>(.*?)</finish>',       r, re.DOTALL).group(1).strip()
        return bool(reasoning) and bool(assess) and finish == 'over'

    return False

ROUND_WEIGHTS = {
    'first':  0.15,
    'middle': 0.25,
    'last':   0.35,
}

def format_reward(responses: list):
    if not responses or len(responses) == 0:
        return 0.0, 0

    n = len(responses)
    weighted_sum = 0.0
    paint_count = 0

    for i, resp in enumerate(responses):
        is_first = (i == 0)
        is_last = (i == n - 1)

        if is_first and is_last:
            kind = 'first'
        elif is_first:
            kind = 'first'
        elif is_last:
            kind = 'last'
        else:
            kind = 'middle'

        if _check_round(resp, kind):
            weighted_sum += ROUND_WEIGHTS[kind]
            if kind in ('first', 'middle'):
                paint_count += 1

    format_score = min(weighted_sum, 1.0)
    return format_score, paint_count

def compute_round_score(image_path, caption, gt_path, data_source, extra_info=None):
    if not image_path or not os.path.exists(str(image_path)):
        return 0.0, {}

    extra_info = extra_info or {}
    weights = get_weights(data_source)

    score = 0.0
    details = {}

    if "clip_t" in weights and caption:
        v = compute_clip_t(image_path, caption)
        details["clip_t"] = v
        score += weights["clip_t"] * v

    if "hps" in weights and caption:
        v = compute_hps_score(image_path, caption)
        details["hps"] = v
        score += weights["hps"] * v

    if "aesthetic" in weights:
        v = llm_aesthetic_judge(image_path)
        details["aesthetic"] = v
        score += weights["aesthetic"] * v

    if "gdino" in weights and caption:
        v = compute_gdino_score(image_path, caption)
        details["gdino"] = v
        score += weights["gdino"] * v

    if "ocr" in weights and caption:
        v = glm_ocr_judge(image_path, caption, data_source)
        details["ocr"] = v
        score += weights["ocr"] * v

    return score, details

def compute_monotonic_ratio(round_scores):
    if len(round_scores) <= 1:
        return 0.0

    for t in range(1, len(round_scores)):
        if round_scores[t] <= round_scores[t - 1]:
            return 0.0

    s0 = round_scores[0]
    total_gain = round_scores[-1] - s0
    return min(total_gain / max(1.0 - s0, 1e-6), 1.0)

def compute_content_reward(round_scores):
    if not round_scores:
        return 0.0

    final_score = round_scores[-1]
    monotonic_ratio = compute_monotonic_ratio(round_scores)
    return 0.8 * final_score + 0.2 * monotonic_ratio

def compute_score(prediction: list, ground_truth: list,
                  image_url=None, caption=None, extra_info=None,
                  image_urls_per_round=None):
    extra_info = extra_info or {}
    data_source = extra_info.get("data_source", "default")

    assert len(prediction) > 0, "[draw_score] Model responses are empty!"

    format_score, paint_count = format_reward(prediction)

    if format_score != 1.0:
        reward = -1.0 + format_score
        extra_info['_draw_metrics'] = {
            'format_score': format_score,
            'paint_count': paint_count,
            'round_scores': [],
            'monotonic_ratio': 0.0,
            'final_round_score': 0.0,
            'content_reward': 0.0,
            'reward': reward,
        }
        print(
            f"[draw_score] Format not perfect: format_score={format_score:.3f} "
            f"→ reward={reward:.4f} (skipping content evaluation)"
        )
        return reward

    if image_urls_per_round and isinstance(image_urls_per_round, list) and len(image_urls_per_round) > 0:
        round_images = [p for p in image_urls_per_round if p and os.path.exists(str(p))]
    elif image_url and os.path.exists(str(image_url)):
        round_images = [image_url]
    else:
        print(f"[draw_score] No images available | format_score={format_score:.3f}")
        return 0.0

    if not caption:
        print(f"[draw_score] No caption available | format_score={format_score:.3f}")
        return 0.0

    gt_path = ground_truth[0] if ground_truth else None

    round_scores = []
    round_details_list = []
    for r_idx, r_image in enumerate(round_images):
        s, details = compute_round_score(r_image, caption, gt_path, data_source, extra_info)
        round_scores.append(s)
        round_details_list.append(details)
        print(f"[draw_score] Round {r_idx}: image={r_image} score={s:.4f} details={details}")

    content_reward = compute_content_reward(round_scores)
    monotonic_ratio = compute_monotonic_ratio(round_scores) if len(round_scores) > 1 else 0.0
    reward = content_reward

    final_details = round_details_list[-1] if round_details_list else {}
    extra_info['_draw_metrics'] = {
        'format_score': format_score,
        'paint_count': paint_count,
        'round_scores': round_scores,
        'monotonic_ratio': monotonic_ratio,
        **{f'final_{k}': v for k, v in final_details.items()},
        'final_round_score': round_scores[-1] if round_scores else 0.0,
        'content_reward': content_reward,
        'reward': reward,
    }

    print(
        f"[draw_score] round_scores={[f'{s:.4f}' for s in round_scores]} | "
        f"final={round_scores[-1]:.4f} | monotonic={monotonic_ratio:.3f} | "
        f"content={content_reward:.4f} | format={format_score:.3f} | "
        f"paint_count={paint_count} | reward={reward:.4f}"
    )
    return reward

def compute_scores_batch(
    predictions: list,
    ground_truths: list,
    image_urls: list,
    captions: list,
    extra_infos: list,
    image_urls_per_round_list: list = None,
) -> list:
    n = len(predictions)
    extra_infos = [e or {} for e in extra_infos]
    if image_urls_per_round_list is None:
        image_urls_per_round_list = [None] * n

    format_scores  = []
    paint_counts   = []
    for i in range(n):
        fs, pc = format_reward(predictions[i])
        format_scores.append(fs)
        paint_counts.append(pc)

    final_scores = [None] * n
    for i in range(n):
        if format_scores[i] != 1.0:
            final_scores[i] = -1.0 + format_scores[i]
            extra_infos[i]['_draw_metrics'] = {
                'format_score': format_scores[i],
                'paint_count': paint_counts[i],
                'round_scores': [],
                'monotonic_ratio': 0.0,
                'final_round_score': 0.0,
                'content_reward': 0.0,
                'reward': final_scores[i],
            }

    valid_idx = []
    valid_round_images = []
    valid_round_counts = []
    valid_captions = []
    valid_data_sources = []
    valid_gt_paths = []

    for i in range(n):
        if final_scores[i] is not None:
            continue

        per_round = image_urls_per_round_list[i]
        if per_round and isinstance(per_round, list) and len(per_round) > 0:
            imgs = [p for p in per_round if p and os.path.exists(str(p))]
        elif image_urls[i] and os.path.exists(str(image_urls[i])):
            imgs = [image_urls[i]]
        else:
            imgs = []

        if not imgs or not captions[i]:
            final_scores[i] = 0.0
            continue

        valid_idx.append(i)
        valid_round_images.append(imgs)
        valid_round_counts.append(len(imgs))
        valid_captions.append(captions[i])
        valid_data_sources.append(extra_infos[i].get("data_source", "default"))
        gt = ground_truths[i]
        valid_gt_paths.append(gt[0] if gt else None)

    if not valid_idx:
        return [s if s is not None else 0.0 for s in final_scores]

    all_images = []
    all_captions_flat = []
    all_data_sources_flat = []
    for j in range(len(valid_idx)):
        imgs = valid_round_images[j]
        cap = valid_captions[j]
        ds = valid_data_sources[j]
        all_images.extend(imgs)
        all_captions_flat.extend([cap] * len(imgs))
        all_data_sources_flat.extend([ds] * len(imgs))

    total_images = len(all_images)
    print(f"[draw_score batch] {len(valid_idx)} valid samples, {total_images} total round images")

    all_needed_metrics = set()
    per_sample_weights = []
    for j in range(len(valid_idx)):
        w = get_weights(valid_data_sources[j])
        per_sample_weights.append(w)
        all_needed_metrics.update(w.keys())

    metric_scores = {}

    if "clip_t" in all_needed_metrics:
        metric_scores["clip_t"] = compute_clip_t_batch(all_images, all_captions_flat)

    if "hps" in all_needed_metrics:
        metric_scores["hps"] = compute_hps_score_batch(all_images, all_captions_flat)

    if "aesthetic" in all_needed_metrics:
        metric_scores["aesthetic"] = compute_llm_aesthetic_batch(all_images)

    if "gdino" in all_needed_metrics:
        metric_scores["gdino"] = compute_gdino_batch(all_images, all_captions_flat)

    if "ocr" in all_needed_metrics:
        metric_scores["ocr"] = compute_llm_ocr_batch(
            all_images, all_captions_flat, all_data_sources_flat
        )

    offset = 0
    for j, i in enumerate(valid_idx):
        cnt = valid_round_counts[j]
        weights = per_sample_weights[j]

        round_scores = []
        round_details_list = []
        for k in range(cnt):
            flat_idx = offset + k
            s = 0.0
            details = {}
            for metric_name, w in weights.items():
                if metric_name in metric_scores:
                    v = metric_scores[metric_name][flat_idx]
                    details[metric_name] = v
                    s += w * v
            round_scores.append(s)
            round_details_list.append(details)
        offset += cnt

        content_reward = compute_content_reward(round_scores)
        monotonic_ratio = compute_monotonic_ratio(round_scores) if len(round_scores) > 1 else 0.0

        final_scores[i] = content_reward

        final_details = round_details_list[-1] if round_details_list else {}
        extra_infos[i]['_draw_metrics'] = {
            'format_score': format_scores[i],
            'paint_count': paint_counts[i],
            'round_scores': round_scores,
            'monotonic_ratio': monotonic_ratio,
            **{f'final_{k}': v for k, v in final_details.items()},
            'final_round_score': round_scores[-1] if round_scores else 0.0,
            'content_reward': content_reward,
            'reward': final_scores[i],
        }

        print(
            f"[draw_score batch] idx={i} rounds={cnt} "
            f"round_scores={[f'{s:.4f}' for s in round_scores]} | "
            f"content={content_reward:.4f} | "
            f"format={format_scores[i]:.3f} | reward={final_scores[i]:.4f}"
        )

    return [s if s is not None else 0.0 for s in final_scores]