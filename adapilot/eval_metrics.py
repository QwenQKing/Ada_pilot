import pandas as pd
import argparse
import functools
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.abspath('.'))

from adapilot.utils.reward_score_mm.draw_score import (
    normalize_clip_t,
    compute_clip_t_batch,
    compute_hps_score_batch,
    compute_llm_ocr_batch,
    compute_gdino_score,
    llm_aesthetic_judge      as _vlm_aesthetic_judge,
)

KIMI_DEFAULT_HOST  = os.environ.get("KIMI_API_BASE", "")
KIMI_DEFAULT_MODEL = "kimi-k2.5"
KIMI_CHAT_PATH     = "/v1/chat/completions"

def _kimi_url() -> str:
    custom = os.environ.get("KIMI_API_BASE")
    if custom:
        host = custom.rstrip("/")
        if not host.startswith("http"):
            host = "http://" + host
        if host.endswith("/v1"):
            return host + "/chat/completions"
        return host + "/v1/chat/completions"
    return ""

@functools.lru_cache(maxsize=1)
def _kimi_token() -> str:
    tok = os.environ.get("OPENAI_API_KEY") or os.environ.get("KIMI_TOKEN")
    if tok:
        return tok.strip()
    for p in ("kimi_token.txt", "api_token.txt", os.path.expanduser("~/kimi_token.txt")):
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip().lstrip("﻿")
                    if line:
                        return line
    return ""

def _kimi_headers() -> dict:
    tok = _kimi_token()
    return {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {tok}",
    }

def _kimi_extract(data: dict) -> str:
    try:
        msg = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return ""
    return (msg.get("content") or msg.get("reasoning_content") or "").strip()

_KIMI_ENTITY_CACHE = None
_KIMI_OCR_GT_CACHE = None
_KIMI_AESTHETIC_CACHE = None
_KIMI_RP_CACHE = None

def _kimi_image_cache_key(img_path):
    norm = os.path.abspath(img_path).replace("\\", "/")
    parts = norm.split("/final-table3-results/")
    if len(parts) >= 2:
        method = parts[-1].split("/")[0]
        return f"{method}/{os.path.basename(img_path)}"
    return os.path.basename(img_path)

def _load_kimi_caches():
    global _KIMI_ENTITY_CACHE, _KIMI_OCR_GT_CACHE
    global _KIMI_AESTHETIC_CACHE, _KIMI_RP_CACHE
    if _KIMI_ENTITY_CACHE is not None:
        return
    cache_dir = os.environ.get("KIMI_CACHE_DIR", "")
    _KIMI_ENTITY_CACHE = {}
    _KIMI_OCR_GT_CACHE = {}
    _KIMI_AESTHETIC_CACHE = {}
    _KIMI_RP_CACHE = {}
    if not cache_dir or not os.path.isdir(cache_dir):
        return
    ent_fp = os.path.join(cache_dir, "entities.json")
    ocr_fp = os.path.join(cache_dir, "ocr_gt.json")
    ae_fp  = os.path.join(cache_dir, "aesthetic.json")
    rp_fp  = os.path.join(cache_dir, "realism_physics.json")
    if os.path.isfile(ent_fp):
        try:
            with open(ent_fp, encoding="utf-8") as f:
                _KIMI_ENTITY_CACHE = json.load(f)
            print(f"[kimi_cache] entities loaded: {len(_KIMI_ENTITY_CACHE)} entries")
        except Exception as e:
            print(f"[kimi_cache] load entities failed: {e}")
    if os.path.isfile(ocr_fp):
        try:
            with open(ocr_fp, encoding="utf-8") as f:
                _KIMI_OCR_GT_CACHE = json.load(f)
            print(f"[kimi_cache] ocr_gt   loaded: {len(_KIMI_OCR_GT_CACHE)} entries")
        except Exception as e:
            print(f"[kimi_cache] load ocr_gt failed: {e}")
    if os.path.isfile(ae_fp):
        try:
            with open(ae_fp, encoding="utf-8") as f:
                _KIMI_AESTHETIC_CACHE = json.load(f)
            print(f"[kimi_cache] aesthetic loaded: {len(_KIMI_AESTHETIC_CACHE)} entries")
        except Exception as e:
            print(f"[kimi_cache] load aesthetic failed: {e}")
    if os.path.isfile(rp_fp):
        try:
            with open(rp_fp, encoding="utf-8") as f:
                _KIMI_RP_CACHE = json.load(f)
            print(f"[kimi_cache] realism_physics loaded: {len(_KIMI_RP_CACHE)} entries")
        except Exception as e:
            print(f"[kimi_cache] load realism_physics failed: {e}")

def _use_kimi() -> bool:
    return os.environ.get("USE_KIMI", "0").strip().lower() in ("1", "true", "yes", "on")

def _kimi_extract_entities(caption: str) -> list:
    import json as _json
    import re as _re
    import time
    import requests

    _load_kimi_caches()
    if caption in _KIMI_ENTITY_CACHE:
        cached = _KIMI_ENTITY_CACHE[caption]
        if isinstance(cached, list):
            return cached

    url   = _kimi_url()
    model = os.environ.get("KIMI_MODEL", KIMI_DEFAULT_MODEL).strip() or KIMI_DEFAULT_MODEL
    max_retries = int(os.environ.get("VLM_ENTITY_MAX_RETRIES", 5))

    if not _kimi_token():
        print("[extract_entities] no KIMI_TOKEN / kimi_token.txt / api_token.txt, return []")
        return []

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

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4096,
        "n": 1,
        "enable_thinking": False,
        "chat_template_kwargs": {"thinking": False},
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=_kimi_headers(), json=payload, timeout=120)
            if resp.status_code == 503:
                wait = 5 * (attempt + 1)
                print(f"[extract_entities] Server busy, retry {attempt+1}/{max_retries} after {wait}s")
                if attempt < max_retries - 1:
                    time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()

            if isinstance(data, dict) and data.get("success") is False:
                print(f"[extract_entities] gateway: {data.get('status_msg')}")
                if attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))
                    continue
                return []

            content = _kimi_extract(data)
            if not content:
                fr = data.get("choices", [{}])[0].get("finish_reason")
                print(f"[extract_entities] empty content (finish_reason={fr})")
                return []

            start = content.find('[')
            end   = content.rfind(']') + 1
            if start == -1 or end == 0:
                print(f"[extract_entities] No JSON array found: {content[:200]}")
                return []

            raw_list = _json.loads(content[start:end])
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
                label = _re.sub(r'["\'].*?["\']', '', label).strip()
                label = _re.sub(r'\s+', ' ', label).strip()
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

from adapilot.utils.reward_score_mm import draw_score as _draw_score_mod
_vlm_extract_entities = _draw_score_mod.extract_entities_with_llm

def _dispatch_extract_entities(caption: str) -> list:
    if _use_kimi():
        return _kimi_extract_entities(caption)
    return _vlm_extract_entities(caption)

_draw_score_mod.extract_entities_with_llm = _dispatch_extract_entities

def _kimi_extract_ocr_gt(caption: str) -> str:
    import re as _re
    import time
    import requests

    _load_kimi_caches()
    if caption in _KIMI_OCR_GT_CACHE:
        cached = _KIMI_OCR_GT_CACHE[caption]
        if isinstance(cached, str):
            return cached

    if not _kimi_token():
        print("[kimi_ocr_gt] no KIMI_TOKEN / kimi_token.txt / api_token.txt")
        return ""

    url   = _kimi_url()
    model = os.environ.get("KIMI_MODEL", KIMI_DEFAULT_MODEL).strip() or KIMI_DEFAULT_MODEL
    max_retries = int(os.environ.get("VLM_ENTITY_MAX_RETRIES", 5))

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

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": gt_prompt}],
        "max_tokens": 4096,
        "n": 1,
        "enable_thinking": False,
        "chat_template_kwargs": {"thinking": False},
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=_kimi_headers(), json=payload, timeout=120)
            if resp.status_code == 503:
                wait = 5 * (attempt + 1)
                print(f"[kimi_ocr_gt] Server busy, retry {attempt+1}/{max_retries} after {wait}s")
                if attempt < max_retries - 1:
                    time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()

            if isinstance(data, dict) and data.get("success") is False:
                print(f"[kimi_ocr_gt] gateway: {data.get('status_msg')}")
                if attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))
                    continue
                return ""

            content = _kimi_extract(data)
            if not content:
                fr = data.get("choices", [{}])[0].get("finish_reason")
                print(f"[kimi_ocr_gt] empty content (finish_reason={fr})")
                return ""

            content = content.replace("<|user|>", "").replace("<|assistant|>", "").strip()
            match = _re.search(r'<text>(.*?)</text>', content, _re.DOTALL)
            if match:
                gt_text = match.group(1).strip()
                print(f"[kimi_ocr_gt] GT text extracted: '{gt_text[:80]}'")
                return gt_text
            print(f"[kimi_ocr_gt] No <text> tag found in response: {content[:200]}")
            return ""

        except requests.exceptions.Timeout:
            wait = 5 * (attempt + 1)
            print(f"[kimi_ocr_gt] Timeout, retry {attempt+1}/{max_retries} after {wait}s")
            if attempt < max_retries - 1:
                time.sleep(wait)
        except Exception as e:
            wait = 5 * (attempt + 1)
            print(f"[kimi_ocr_gt] Failed ({e}), retry {attempt+1}/{max_retries} after {wait}s")
            if attempt < max_retries - 1:
                time.sleep(wait)

    print("[kimi_ocr_gt] All retries failed")
    return ""

def _call_glm_ocr_api(gen_b64: str, api_bases, model: str, prefix: str = "glm_ocr", max_retries: int = 4) -> str:
    import requests
    import time as _time
    if isinstance(api_bases, str):
        api_bases = [api_bases]
    n = len(api_bases)
    for attempt in range(max_retries):
        api_base = api_bases[attempt % n]
        next_base = api_bases[(attempt + 1) % n]
        try:
            resp = requests.post(
                f"{api_base}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": model,
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
                print(f"[{prefix}] ({api_base}) Server busy, retry {attempt+1}/{max_retries} after {wait}s → next: {next_base}")
                _time.sleep(wait)
                continue
            resp.raise_for_status()
            ocr_text = resp.json()["choices"][0]["message"]["content"].strip()
            ocr_text = ocr_text.replace("<|user|>", "").replace("<|assistant|>", "").strip()
            print(f"[{prefix}] ({api_base}) OCR recognized: '{ocr_text[:80]}'")
            return ocr_text
        except requests.exceptions.Timeout:
            wait = 5 * (attempt + 1)
            print(f"[{prefix}] ({api_base}) timeout, retry {attempt+1}/{max_retries} after {wait}s → next: {next_base}")
            if attempt < max_retries - 1:
                _time.sleep(wait)
        except Exception as e:
            print(f"[{prefix}] ({api_base}) failed: {e} → next: {next_base}")
    return ""

def _kimi_glm_ocr_judge(generated_image_path: str, caption: str, data_source: str = "") -> float:
    import base64

    glm_api_base       = os.environ.get("GLM_OCR_API_BASE",       "")
    glm_fallback_base  = os.environ.get("GLM_OCR_FALLBACK_BASE",  "")
    glm_fallback_base2 = os.environ.get("GLM_OCR_FALLBACK_BASE_2", "")
    glm_model          = os.environ.get("GLM_OCR_MODEL", "glm-ocr")
    glm_endpoints = [glm_api_base]
    if glm_fallback_base:  glm_endpoints.append(glm_fallback_base)
    if glm_fallback_base2: glm_endpoints.append(glm_fallback_base2)

    if data_source == "ShowTable":
        gt_text = _draw_score_mod._normalize_text(caption)
        print(f"[kimi_glm_ocr_judge] ShowTable: using caption directly as GT: '{gt_text[:80]}'")
        if not gt_text:
            print("[kimi_glm_ocr_judge] ShowTable: empty caption → OCR score = 1.0")
            return 1.0
    else:
        gt_text = _kimi_extract_ocr_gt(caption)
        if gt_text == "":
            print("[kimi_glm_ocr_judge] GT extraction failed → return None (will retry)")
            return None
        if gt_text.lower() == "none":
            print("[kimi_glm_ocr_judge] No text expected in caption → OCR score = 1.0")
            return 1.0

    try:
        with open(generated_image_path, "rb") as f:
            gen_b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"[kimi_glm_ocr_judge] Failed to encode image: {e} → None")
        return None

    glm_ocr_retries = int(os.environ.get("GLM_OCR_MAX_RETRIES", 6))
    ocr_text = _call_glm_ocr_api(
        gen_b64, glm_endpoints, glm_model,
        prefix="kimi_glm_ocr_judge", max_retries=glm_ocr_retries,
    )

    if not ocr_text:
        print("[kimi_glm_ocr_judge] OCR text empty after all attempts → None (will retry)")
        return None

    metrics = _draw_score_mod._compute_text_metrics(gt_text, ocr_text)
    f_score = metrics["f_score"]
    print(
        f"[kimi_glm_ocr_judge] GT='{gt_text[:60]}' | OCR='{ocr_text[:60]}' | "
        f"C={metrics['C']} I={metrics['I']} D={metrics['D']} S={metrics['S']} | "
        f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} "
        f"f_score={f_score:.4f}"
    )
    return f_score

_vlm_glm_ocr_judge = _draw_score_mod.glm_ocr_judge

def _dispatch_glm_ocr_judge(generated_image_path: str, caption: str, data_source: str = "") -> float:
    if _use_kimi():
        return _kimi_glm_ocr_judge(generated_image_path, caption, data_source)
    return _vlm_glm_ocr_judge(generated_image_path, caption, data_source)

_draw_score_mod.glm_ocr_judge = _dispatch_glm_ocr_judge

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path',    type=str, required=True,
                        help='Path to val_raw*.json')
    parser.add_argument('--output_path',  type=str, default=None,
                        help='Output path for eval results (default: same dir as json_path)')
    parser.add_argument('--workers',      type=int, default=8,
                        help='Concurrency for LLM calls (CLIP/HPS use batch, unaffected)')
    parser.add_argument('--use_llm_ocr',  action='store_true',
                        help='Use GLM-OCR / VLM for OCR (default: EasyOCR)')
    parser.add_argument('--use_gdino',    action='store_true',
                        help='Enable GroundingDINO object evaluation (requires GDINO_MODEL_PATH)')
    parser.add_argument('--use_aesthetic', action='store_true',
                        help='Enable VLM aesthetic evaluation (requires VLM_API_BASE)')
    parser.add_argument('--use_realism_physics', action='store_true',
                        help='Enable realism and physics VLM evaluation')
    parser.add_argument('--project_root', type=str, default=None,
                        help='Project root for resolving image_url relative paths (default: cwd)')
    return parser.parse_args()

def resolve_path(rel_or_abs, project_root: str):
    if rel_or_abs is None:
        return None
    if os.path.isabs(str(rel_or_abs)):
        return str(rel_or_abs)
    return os.path.join(project_root, str(rel_or_abs))

def print_env():
    keys = [
        'CLIP_MODEL_PATH', 'CLIP_SERVER_URL',
        'HPS_MODEL_DIR',   'HPS_SERVER_URL',
        'EASYOCR_MODEL_DIR',
        'GDINO_MODEL_PATH', 'GDINO_SERVER_URL',
        'GLM_OCR_API_BASE', 'GLM_OCR_MODEL',
        'VLM_API_BASE', 'VLM_MODEL',
    ]
    for k in keys:
        print(f'[eval] {k:<26s} = {os.environ.get(k, "")}')

def compute_gdino_batch(image_paths: list, captions: list) -> list:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    gdino_server = os.environ.get("GDINO_SERVER_URL", "")
    if _use_kimi() and os.environ.get("KIMI_MAX_WORKERS"):
        max_workers = int(os.environ["KIMI_MAX_WORKERS"])
    else:
        max_workers = int(os.environ.get("GDINO_MAX_WORKERS", 4))

    results = [0.0] * len(image_paths)
    valid_idx = [
        i for i, (p, c) in enumerate(zip(image_paths, captions))
        if p and os.path.exists(str(p)) and c
    ]

    if not valid_idx:
        print('[eval] GDINO: no valid samples, skip.')
        return results

    if gdino_server:
        print(f'[eval] GDINO remote: {gdino_server}, workers={max_workers}, n={len(valid_idx)}')
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(compute_gdino_score, image_paths[i], captions[i]): i
                for i in valid_idx
            }
            total = len(future_to_idx)
            completed = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = float(future.result())
                    print(f'[eval] GDINO [{idx}] = {results[idx]:.4f}')
                except Exception as e:
                    print(f'[eval] GDINO [{idx}] failed: {e}, result=0.0')
                completed += 1
                if total >= 10 and (completed % 10 == 0 or completed == total):
                    print(f'[eval] GDINO progress: {completed}/{total}')

    else:
        print(f'[eval] GDINO local (serial), n={len(valid_idx)}')
        for rank, i in enumerate(valid_idx, 1):
            try:
                results[i] = float(compute_gdino_score(image_paths[i], captions[i]))
                print(f'[eval] GDINO [{rank}/{len(valid_idx)}] idx={i} = {results[i]:.4f}')
            except Exception as e:
                print(f'[eval] GDINO [{rank}/{len(valid_idx)}] idx={i} local failed: {e}, result=0.0')

    return results

def llm_realism_physics_judge(image_path: str, caption: str) -> dict:
    import base64
    import time
    import requests

    _empty = {"realism": None, "physics": None,
              "realism_reason": "", "physics_reason": "",
              "realism_raw": None, "physics_raw": None}

    _load_kimi_caches()
    if _KIMI_RP_CACHE:
        key = _kimi_image_cache_key(image_path)
        v = _KIMI_RP_CACHE.get(key)
        if v is None and "/" in key:
            v = _KIMI_RP_CACHE.get(os.path.basename(image_path))
        if isinstance(v, dict) and (v.get("realism") is not None or v.get("realism_raw") is not None):
            return v

    is_kimi = _use_kimi()
    if is_kimi:
        if not _kimi_token():
            print("[llm_realism_physics_judge] no KIMI_TOKEN / kimi_token.txt / api_token.txt")
            return _empty
        url     = _kimi_url()
        model   = os.environ.get("KIMI_MODEL", KIMI_DEFAULT_MODEL).strip() or KIMI_DEFAULT_MODEL
        headers = _kimi_headers()
        timeout = 120
    else:
        api_base = os.environ.get("VLM_API_BASE", "").strip()
        model    = os.environ.get("VLM_MODEL", "").strip()
        if not api_base:
            print("[llm_realism_physics_judge] VLM_API_BASE not set")
            return _empty
        url     = f"{api_base}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        timeout = 60

    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"[llm_realism_physics_judge] Failed to encode image: {e}")
        return _empty

    prompt = f"""You are an extremely discerning image quality judge specializing in physical realism and plausibility. Scrutinize every visual detail with ruthless precision. Always look for even subtle flaws: AI generation artifacts, physically impossible lighting, incorrect material behavior, violated spatial relationships, or any detail that betrays artificial generation. Be wary of awarding high scores unless the image truly meets the strictest standards.

Return ONLY the final JSON object based on these internal instructions:

PROMPT used to generate this image: {caption}

## Realism Score (1-5)
Evaluate how realistically the image is rendered, independent of the prompt. Focus purely on visual credibility.

1 Point:
- Obvious AI artifacts: distorted fingers, malformed text, fused or melted faces/objects.
- Fundamental physics violations: objects floating without support, impossible shadow directions.
- Completely wrong material representation: opaque glass, non-reflective metal.
- Severely distorted spatial relationships: objects clipping, wildly inconsistent scale.

2 Points:
- Multiple noticeable flaws that immediately betray AI generation.
- Physical relationships mostly recognizable but with 1-2 significant errors.
- Materials clearly synthetic or rough; lighting flat or incorrect.
- Shadows harsh, inconsistent, or misaligned with implied light source.

3 Points:
- Visually plausible to a casual observer; flaws require closer inspection.
- Physical relationships follow common sense: support, gravity, occlusion are correct.
- Materials show reasonable surface properties; lighting broadly consistent.
- Minor issues remain: slight edge bleeding, locally unnatural textures.

4 Points:
- Near-photorealistic; flaws only detectable under careful scrutiny.
- Physically accurate: soft shadows, plausible reflections/refractions, correct subsurface scattering.
- Materials convincingly rendered with appropriate texture, gloss, and lighting response.
- Barely perceptible AI traces that do not disrupt overall realism.

5 Points:
- Rarely awarded; indistinguishable from a real photograph under normal viewing.
- All physical laws perfectly obeyed: lighting, shadow, reflection, refraction, gravity, material behavior.
- Zero AI artifacts detectable even under pixel-level inspection.
- Reserved exclusively for images genuinely mistakable for a real photograph.

## Physics Score (1-5 or N/A)
Evaluate whether physical constraints described or implied by the PROMPT are correctly realized.
If the PROMPT contains no verifiable physical constraints (abstract art, pure texture, no spatial relationships), output Physics score as "N/A".

1 Point:
- Directly contradicts physical constraints stated in the prompt.
- Supported/stacked objects are floating, clipping, or detached.
- Light source direction stated in prompt is completely wrong in the image.
- Material properties specified in prompt show none of the expected physical behavior.

2 Points:
- Multiple physical constraints from the prompt are violated or poorly realized.
- Spatial relationships (above, below, inside, in front of) mostly incorrect.
- Shadow and lighting inconsistent with the prompt's described environment.
- Physical interactions (pouring, falling, reflecting) visually implausible.

3 Points:
- Most physical constraints from the prompt are correctly realized.
- Primary spatial relationships correct; secondary relationships may have minor errors.
- Lighting environment broadly matches the prompt's description.
- Physical interactions believable, though not perfectly accurate in detail.

4 Points:
- All explicitly stated physical constraints correctly realized.
- Spatial relationships accurate and consistent throughout the image.
- Lighting, shadow direction, and environment precisely match the prompt.
- Material physics (transparency, reflectivity, deformation) correctly portrayed.

5 Points:
- Rarely awarded; every physical constraint — explicit and implied — flawlessly realized.
- Emergent physical details not stated in prompt also correctly present (secondary reflections, contact shadows, material deformation under weight).
- No physical violation detectable even under rigorous analysis.
- Reserved for images serving as ground-truth references for the physical scenario described.

YOU MUST output ONLY the following JSON, nothing else, no markdown, no explanation, no step labels:
{{"realism_reason": "...", "realism_score": int, "physics_reason": "...", "physics_score": int or "N/A"}}
"""

    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": prompt},
            ],
        }],
        "max_tokens": 4096,
        "n": 1,
        "enable_thinking": False,
        "chat_template_kwargs": {"thinking": False},
    }

    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                url,
                headers=_kimi_headers(),
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()

            if isinstance(data, dict) and data.get("success") is False:
                raise ValueError(f"gateway: {data.get('status_msg')}")

            content = _kimi_extract(data)
            if not content:
                fr = data.get("choices", [{}])[0].get("finish_reason")
                raise ValueError(f"empty content (finish_reason={fr}, usage={data.get('usage')})")

            content = content.replace("```json", "").replace("```", "").strip()

            json_start = content.find('{')
            json_end   = content.rfind('}') + 1
            if json_start == -1 or json_end == 0:
                raise ValueError(f"No JSON found in response: {content[:200]}")

            result = json.loads(content[json_start:json_end])

            realism_raw    = result.get("realism_score")
            physics_raw    = result.get("physics_score", "N/A")
            realism_reason = result.get("realism_reason", "")
            physics_reason = result.get("physics_reason", "")

            realism_norm = float(max(0.0, min(1.0, (float(realism_raw) - 1.0) / 4.0))) \
                           if realism_raw not in (None, "N/A") else None
            physics_norm = None if physics_raw in (None, "N/A") else \
                           float(max(0.0, min(1.0, (float(physics_raw) - 1.0) / 4.0)))

            print(
                f"[llm_realism_physics_judge] "
                f"realism={realism_raw}({realism_norm:.3f} normalized) | "
                f"physics={physics_raw}"
                + (f"({physics_norm:.3f} normalized)" if physics_norm is not None else "(N/A)")
            )
            return {
                "realism":        realism_norm,
                "physics":        physics_norm,
                "realism_reason": realism_reason,
                "physics_reason": physics_reason,
                "realism_raw":    realism_raw,
                "physics_raw":    physics_raw,
            }

        except Exception as e:
            print(f"[llm_realism_physics_judge] Attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"[llm_realism_physics_judge] Retrying in {wait}s ...")
                time.sleep(wait)
            else:
                print(f"[llm_realism_physics_judge] All {max_retries} attempts failed, returning None.")
                return _empty
 
def compute_realism_physics_batch(
    image_paths: list,
    captions:    list,
    max_workers: int = 8,
) -> list:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    _empty = {"realism": None, "physics": None,
              "realism_reason": "", "physics_reason": "",
              "realism_raw": None, "physics_raw": None}

    if _use_kimi() and os.environ.get("KIMI_MAX_WORKERS"):
        max_workers = int(os.environ["KIMI_MAX_WORKERS"])

    n       = len(image_paths)
    results = [dict(_empty) for _ in range(n)]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(llm_realism_physics_judge, p, c): i
            for i, (p, c) in enumerate(zip(image_paths, captions))
            if p and os.path.isfile(str(p)) and c
        }
        completed = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"[compute_realism_physics_batch] idx={idx} failed: {e}")
            completed += 1
            if completed % 10 == 0 or completed == len(future_to_idx):
                print(f"[compute_realism_physics_batch] {completed}/{len(future_to_idx)}")
 
    return results

def llm_aesthetic_judge(image_path: str) -> float:
    import base64
    import json as _json
    import time
    import requests

    _load_kimi_caches()
    if _KIMI_AESTHETIC_CACHE:
        key = _kimi_image_cache_key(image_path)
        v = _KIMI_AESTHETIC_CACHE.get(key)
        if v is None and "/" in key:
            v = _KIMI_AESTHETIC_CACHE.get(os.path.basename(image_path))
        if isinstance(v, (int, float)):
            return float(v)

    url   = _kimi_url()
    model = os.environ.get("KIMI_MODEL", KIMI_DEFAULT_MODEL).strip() or KIMI_DEFAULT_MODEL
    max_retries = int(os.environ.get("LLM_AESTHETIC_MAX_RETRIES", 5))

    if not _kimi_token():
        print("[llm_aesthetic_judge] no KIMI_TOKEN / kimi_token.txt / api_token.txt → None")
        return None

    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"[llm_aesthetic_judge] Failed to encode image: {e} → None")
        return None

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
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": prompt},
            ],
        }],
        "max_tokens": 4096,
        "n": 1,
        "enable_thinking": False,
        "chat_template_kwargs": {"thinking": False},
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                url,
                headers=_kimi_headers(),
                json=payload,
                timeout=120,
            )
            if resp.status_code == 503:
                wait = 10 * (attempt + 1)
                print(f"[llm_aesthetic_judge] Server busy, retry {attempt+1}/{max_retries} after {wait}s")
                if attempt < max_retries - 1:
                    time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()

            if isinstance(data, dict) and data.get("success") is False:
                print(f"[llm_aesthetic_judge] gateway failure: {data.get('status_msg')}")
                if attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))
                    continue
                return None

            content = _kimi_extract(data)
            if not content:
                fr = data.get("choices", [{}])[0].get("finish_reason")
                print(f"[llm_aesthetic_judge] empty content (finish_reason={fr}, usage={data.get('usage')}) → None")
                return None

            json_start = content.find('{')
            json_end   = content.rfind('}') + 1
            if json_start == -1 or json_end == 0:
                print(f"[llm_aesthetic_judge] No JSON found: {content[:200]} → None")
                return None

            result = _json.loads(content[json_start:json_end])
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

    print("[llm_aesthetic_judge] All retries failed, returning None (will be excluded from mean, recorded for retry)")
    return None

def compute_llm_aesthetic_batch(image_paths: list, max_workers: int = 8) -> list:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if _use_kimi() and os.environ.get("KIMI_MAX_WORKERS"):
        max_workers = int(os.environ["KIMI_MAX_WORKERS"])

    n       = len(image_paths)
    results = [None] * n

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(llm_aesthetic_judge, p): i
            for i, p in enumerate(image_paths)
            if p and os.path.isfile(str(p))
        }
        completed = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                r = future.result()
                results[idx] = float(r) if r is not None else None
            except Exception as e:
                print(f"[compute_llm_aesthetic_batch] idx={idx} failed: {e} → None")
                results[idx] = None
            completed += 1
            if completed % 10 == 0 or completed == len(future_to_idx):
                print(f"[compute_llm_aesthetic_batch] {completed}/{len(future_to_idx)}")

    return results

def main():
    args = parse_args()
    project_root = args.project_root or os.getcwd()

    json_path = resolve_path(args.json_path, project_root)
    with open(json_path, 'r', encoding='utf-8') as f:
        records = json.load(f)

    print(f'\n[eval] Loaded {len(records)} records from {json_path}')
    print(f'[eval] project_root   = {project_root}')
    print(f'[eval] use_llm_ocr    = {args.use_llm_ocr}')
    print(f'[eval] use_gdino      = {args.use_gdino}')
    print(f'[eval] use_aesthetic  = {args.use_aesthetic}')
    print(f'[eval] workers        = {args.workers}')
    print_env()

    valid_records = []
    invalid_count = 0
    for idx, rec in enumerate(records):
        image_url = resolve_path(
            rec.get('final_image_url') or rec.get('image_url'), project_root
        )
        caption = rec.get('caption', '')
        if image_url and os.path.exists(image_url) and caption:
            valid_records.append((idx, rec, image_url, caption))
        else:
            print(f'[eval] Skip idx={idx}: image_url={image_url}')
            invalid_count += 1

    print(f'\n[eval] Valid: {len(valid_records)}, Skipped: {invalid_count}')
    if not valid_records:
        print('[eval] No valid samples, exit.')
        return

    orig_indices = [v[0] for v in valid_records]
    image_paths  = [v[2] for v in valid_records]
    captions     = [v[3] for v in valid_records]

    n = len(valid_records)

    from concurrent.futures import ThreadPoolExecutor
    print(f'\n[eval] Running CLIP-T + HPS + GDINO in parallel (n={n})')
    with ThreadPoolExecutor(max_workers=3) as _p:
        fut_clip  = _p.submit(compute_clip_t_batch, image_paths, captions)
        fut_hps   = _p.submit(compute_hps_score_batch, image_paths, captions)
        fut_gdino = _p.submit(compute_gdino_batch, image_paths, captions) if args.use_gdino else None

        clip_t_raws = fut_clip.result()
        hps_vals    = fut_hps.result()
        gdino_vals  = fut_gdino.result() if fut_gdino else [None] * n

    clip_t_vals = [normalize_clip_t(v) for v in clip_t_raws]
    print(f'[eval] CLIP-T mean = {np.mean(clip_t_vals):.4f}')
    print(f'[eval] HPS mean    = {np.mean(hps_vals):.4f}')
    if args.use_gdino:
        gdino_valid = [v for v in gdino_vals if v is not None]
        if gdino_valid:
            print(f'[eval] GDINO mean  = {np.mean(gdino_valid):.4f}  (n={len(gdino_valid)})')
        else:
            print('[eval] GDINO: no valid results')

    OCR_DATASETS_LOWER = {"postercraft", "textatlas5m", "finmme", "showtable"}
    _dataset_name = os.path.basename(os.path.dirname(args.json_path))
    if _dataset_name.lower() in OCR_DATASETS_LOWER:
        print(f'\n[eval] ── OCR (llm_ocr, workers={args.workers}) ──')
        ocr_vals = compute_llm_ocr_batch(image_paths, captions, max_workers=args.workers)
        ocr_valid = [v for v in ocr_vals if v is not None]
        if ocr_valid:
            print(f'[eval] OCR mean = {np.mean(ocr_valid):.4f}  (valid={len(ocr_valid)}/{n}, fail={n-len(ocr_valid)})')
        else:
            print(f'[eval] OCR: all failed')
    else:
        print(f'
[eval] OCR skipped (dataset {_dataset_name} does not use OCR score)')
        ocr_vals = [None] * n

    aesthetic_vals = [None] * n
    if args.use_aesthetic:
        print(f'\n[eval] ── Aesthetic VLM (concurrent, workers={args.workers}, n={n}) ──')
        aesthetic_vals = compute_llm_aesthetic_batch(image_paths, max_workers=args.workers)
        aesthetic_valid = [v for v in aesthetic_vals if v is not None]
        print(f'[eval] Aesthetic mean = {np.mean(aesthetic_valid):.4f}  (n={len(aesthetic_valid)})')

    realism_physics_vals = [{"realism": None, "physics": None,
                          "realism_raw": None, "physics_raw": None,
                          "realism_reason": "", "physics_reason": ""}] * n
    if args.use_realism_physics:
        print(f'\n[eval] ── Realism & Physics VLM (concurrent, workers={args.workers}, n={n}) ──')
        realism_physics_vals = compute_realism_physics_batch(
            image_paths, captions, max_workers=args.workers
        )
        r_valid = [v['realism'] for v in realism_physics_vals if v.get('realism') is not None]
        p_valid = [v['physics'] for v in realism_physics_vals if v.get('physics') is not None]
        print(f'[eval] Realism mean = {np.mean(r_valid):.4f}  (n={len(r_valid)})')
        print(f'[eval] Physics mean = {np.mean(p_valid):.4f}  (n={len(p_valid)})')

    per_sample = []
    failed_samples = []
    for j, (orig_idx, rec, image_path, caption) in enumerate(valid_records):
        entry = {
            'orig_idx':        orig_idx,
            'data_source':     rec.get('data_source', 'unknown'),
            'final_image_url': rec.get('final_image_url') or rec.get('image_url'),
            'caption':         caption,
            'clip_t':          round(clip_t_vals[j],  4),
            'clip_t_raw':      round(clip_t_raws[j],  4),
            'hps':             round(hps_vals[j],      4),
            'ocr':             (round(float(ocr_vals[j]), 4) if ocr_vals[j] is not None else None),
        }
        missing = []
        if entry['ocr'] is None:
            missing.append('ocr')
        if args.use_gdino:
            entry['gdino'] = round(float(gdino_vals[j]), 4) if gdino_vals[j] is not None else None
            if entry['gdino'] is None: missing.append('gdino')
        if args.use_aesthetic:
            entry['aesthetic'] = round(float(aesthetic_vals[j]), 4) if aesthetic_vals[j] is not None else None
            if entry['aesthetic'] is None: missing.append('aesthetic')
        if args.use_realism_physics:
            rp = realism_physics_vals[j]
            entry['realism']        = rp.get('realism')
            entry['physics']        = rp.get('physics')
            entry['realism_raw']    = rp.get('realism_raw')
            entry['physics_raw']    = rp.get('physics_raw')
            entry['realism_reason'] = rp.get('realism_reason', '')
            entry['physics_reason'] = rp.get('physics_reason', '')
            if entry['realism'] is None: missing.append('realism')
            if entry['physics'] is None: missing.append('physics')
        per_sample.append(entry)
        if missing:
            failed_samples.append({'orig_idx': orig_idx, 'missing': missing})

    source_buckets = defaultdict(list)
    for r in per_sample:
        source_buckets[r['data_source']].append(r)

    def summarize(items, label=''):
        gd_vals  = [x['gdino']     for x in items if x.get('gdino')     is not None] if args.use_gdino    else []
        ae_vals  = [x['aesthetic'] for x in items if x.get('aesthetic') is not None] if args.use_aesthetic else []
        rp_r_vals = [x['realism'] for x in items if x.get('realism') is not None] \
                    if args.use_realism_physics else []
        rp_p_vals = [x['physics'] for x in items if x.get('physics') is not None] \
                    if args.use_realism_physics else []
        ocr_vals_local = [x['ocr'] for x in items if x.get('ocr') is not None]

        result = {
            'count':  len(items),
            'clip_t': round(float(np.mean([x['clip_t'] for x in items])), 4),
            'hps':    round(float(np.mean([x['hps']   for x in items])), 4),
            'ocr':    round(float(np.mean(ocr_vals_local)), 4) if ocr_vals_local else None,
            'ocr_count': len(ocr_vals_local),
        }
        if args.use_gdino:
            result['gdino']       = round(float(np.mean(gd_vals)), 4) if gd_vals else None
            result['gdino_count'] = len(gd_vals)
        if args.use_aesthetic:
            result['aesthetic']       = round(float(np.mean(ae_vals)), 4) if ae_vals else None
            result['aesthetic_count'] = len(ae_vals)
        if args.use_realism_physics:
            result['realism']       = round(float(np.mean(rp_r_vals)), 4) if rp_r_vals else None
            result['realism_count'] = len(rp_r_vals)
            result['physics']       = round(float(np.mean(rp_p_vals)), 4) if rp_p_vals else None
            result['physics_count'] = len(rp_p_vals)

        print(f'\n[eval] === {label or "ALL"} (n={len(items)}) ===')
        print(f'        CLIP-T    = {result["clip_t"]:.4f}')
        print(f'        HPS       = {result["hps"]:.4f}')
        print(f'        OCR       = {result["ocr"]:.4f}  (n={result["ocr_count"]})' if result["ocr"] is not None else f'        OCR       = N/A')
        if args.use_gdino:
            print(f'        GDINO     = {result["gdino"]:.4f}  (n={len(gd_vals)})' if gd_vals else '        GDINO     = N/A')
        if args.use_aesthetic:
            print(f'        Aesthetic = {result["aesthetic"]:.4f}  (n={len(ae_vals)})' if ae_vals else '        Aesthetic = N/A')
        if args.use_realism_physics:
            print(f'        Realism   = {result["realism"]}  (n={len(rp_r_vals)})' if rp_r_vals else '        Realism   = N/A')
            print(f'        Physics   = {result["physics"]}  (n={len(rp_p_vals)})' if rp_p_vals else '        Physics   = N/A')
        return result

    summary = {}
    for source, items in source_buckets.items():
        summary[source] = summarize(items, label=source)

    summary['__all__'] = summarize(per_sample, label='ALL')

    if args.output_path is None:
        base = os.path.splitext(json_path)[0]
        args.output_path = base + '_eval_metrics.json'

    fail_count_per_metric = defaultdict(int)
    for fs in failed_samples:
        for m in fs['missing']:
            fail_count_per_metric[m] += 1
    if failed_samples:
        print(f'
[eval] {len(failed_samples)} samples have None values')
        for m, c in sorted(fail_count_per_metric.items()):
            print(f'        {m:<12s}  fail={c}')
        print(f'        -> see output["failed_samples"] for details')

    output = {
        'summary':       summary,
        'per_sample':    per_sample,
        'invalid_count': invalid_count,
        'failed_samples': failed_samples,
        'failed_count_per_metric': dict(fail_count_per_metric),
        'config': {
            'use_llm_ocr':         args.use_llm_ocr,
            'use_gdino':           args.use_gdino,
            'use_aesthetic':       args.use_aesthetic,
            'use_realism_physics': args.use_realism_physics,
            'workers':             args.workers,
        },
    }

    def sanitize(obj):
        if isinstance(obj, str):
            return obj.encode('utf-8', errors='replace').decode('utf-8')
        elif isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj

    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(sanitize(output), f, ensure_ascii=False, indent=2)
    print(f'\n[eval] Results saved to: {args.output_path}')

    excel_path = os.path.splitext(args.output_path)[0] + '.xlsx'
    summary_rows = []
    for source, s in summary.items():
        row = {'data_source': source}
        row.update(s)
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_excel(excel_path, index=False)
    print(f'[eval] Excel saved to: {excel_path}')

if __name__ == '__main__':
    main()