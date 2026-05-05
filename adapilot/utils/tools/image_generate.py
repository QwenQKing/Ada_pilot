import asyncio
import base64
import time
from io import BytesIO
from typing import Dict, List, Tuple

import aiohttp
from PIL import Image
import os

_DEFAULT_BASE_URL = os.environ.get("IMAGE_GEN_BASE_URL", "")
_DEFAULT_SIZE     = os.environ.get("IMAGE_GEN_SIZE",     "1024x1024")
_DEFAULT_STEPS    = int(os.environ.get("IMAGE_GEN_STEPS",    "50"))
_DEFAULT_GUIDANCE = float(os.environ.get("IMAGE_GEN_GUIDANCE", "4.5"))
_DEFAULT_TIMEOUT  = int(os.environ.get("IMAGE_GEN_TIMEOUT",  "300"))
_DEFAULT_MAX_CONCURRENT = int(os.environ.get("IMAGE_GEN_MAX_CONCURRENT", "16"))
_DEFAULT_RETRIES = int(os.environ.get("IMAGE_GEN_RETRIES", "3"))

async def _decode_image_from_response(
    session: aiohttp.ClientSession,
    api_endpoint: str,
    data: dict,
    timeout: int,
) -> Image.Image:
    item = data["data"][0]
    b64 = item.get("b64_json")
    if b64:
        return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")

    img_url = item.get("url")
    if not img_url:
        raise ValueError(f"Image generation response missing both 'b64_json' and 'url': {item}")

    if img_url.startswith("http://") or img_url.startswith("https://"):
        full_url = img_url
    else:
        base = api_endpoint.rsplit("/v1/images/generations", 1)[0]
        full_url = f"{base}{img_url}"

    async with session.get(
        full_url,
        timeout=aiohttp.ClientTimeout(total=timeout),
    ) as img_resp:
        if img_resp.status != 200:
            err = await img_resp.text()
            raise RuntimeError(f"Fetch image URL failed HTTP {img_resp.status}: {err[:120]}")
        img_bytes = await img_resp.read()
    return Image.open(BytesIO(img_bytes)).convert("RGB")

async def _async_generate_image_batch(
    prompts: List[str],
    base_url: str = _DEFAULT_BASE_URL,
    size: str = _DEFAULT_SIZE,
    num_inference_steps: int = _DEFAULT_STEPS,
    guidance_scale: float = _DEFAULT_GUIDANCE,
    seed: int = None,
    timeout: int = _DEFAULT_TIMEOUT,
    max_concurrent: int = 16,
    show_progress: bool = True,
) -> List[Dict]:
    semaphore = asyncio.Semaphore(max_concurrent)
    api_endpoint = f"{base_url}/v1/images/generations"

    async def _one(prompt, retries=_DEFAULT_RETRIES):
        async with semaphore:
            for attempt in range(retries):
                payload = {
                    "prompt": prompt,
                    "size": size,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                }
                if seed is not None:
                    payload["seed"] = seed
                start = time.time()
                try:
                    async with session.post(
                        api_endpoint,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=timeout),
                    ) as resp:
                        if resp.status != 200:
                            err = await resp.text()
                            print(f"[attempt {attempt+1}/{retries}] HTTP {resp.status}: {err[:80]}")
                            if attempt < retries - 1:
                                await asyncio.sleep(2 ** attempt)
                                continue
                            return {"success": False, "image": None,
                                    "error": f"HTTP {resp.status}: {err}",
                                    "duration": time.time() - start}
                        data = await resp.json()
                        pil_image = await _decode_image_from_response(
                            session, api_endpoint, data, timeout
                        )
                        return {"success": True, "image": pil_image,
                                "error": None, "duration": time.time() - start}

                except asyncio.TimeoutError:
                    print(f"[attempt {attempt+1}/{retries}] Timeout after {timeout}s, prompt: {prompt[:40]}...")
                    if attempt < retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return {"success": False, "image": None,
                            "error": f"Timeout after {timeout}s",
                            "duration": time.time() - start}
                except Exception as e:
                    print(f"[attempt {attempt+1}/{retries}] Error: {e}")
                    if attempt < retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return {"success": False, "image": None,
                            "error": str(e), "duration": time.time() - start}

    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(*[_one(p) for p in prompts])
    return list(results)

def call_image_generate_batch(
    prompts: List[str],
    base_url: str = _DEFAULT_BASE_URL,
    size: str = _DEFAULT_SIZE,
    num_inference_steps: int = _DEFAULT_STEPS,
    guidance_scale: float = _DEFAULT_GUIDANCE,
    seed: int = None,
    timeout: int = _DEFAULT_TIMEOUT,
    max_concurrent: int = _DEFAULT_MAX_CONCURRENT,
) -> List[Tuple[str, List[Image.Image], Dict]]:
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                results = executor.submit(
                    asyncio.run,
                    _async_generate_image_batch(
                        prompts=prompts, base_url=base_url, size=size,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale, seed=seed,
                        timeout=timeout, max_concurrent=max_concurrent,
                    )
                ).result()
        else:
            results = loop.run_until_complete(
                _async_generate_image_batch(
                    prompts=prompts, base_url=base_url, size=size,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale, seed=seed,
                    timeout=timeout, max_concurrent=max_concurrent,
                )
            )
    except RuntimeError:
        results = asyncio.run(
            _async_generate_image_batch(
                prompts=prompts, base_url=base_url, size=size,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale, seed=seed,
                timeout=timeout, max_concurrent=max_concurrent,
            )
        )

    outputs = []
    for prompt, result in zip(prompts, results):
        if result["success"]:
            tool_returned_str = (
                "[Image Generation Results] "
                "The image has been successfully generated based on your prompt. "
                "Here is the generated image:\n"
                "<|vision_start|><|image_pad|><|vision_end|>\n"
                f"prompt used: {prompt}\n"
            )
            tool_returned_images = [result["image"]]
        else:
            print(f"[call_image_generate_batch] failed: {result['error']}")
            tool_returned_str = (
                "[Image Generation Results] "
                "There is an error encountered in generating the image. "
                "Please refine your prompt or reason with your own capabilities.\n"
                f"Error: {result['error']}\n"
            )
            tool_returned_images = []

        tool_stat = {
            "success": result["success"],
            "num_images": len(tool_returned_images),
            "duration": result.get("duration", 0.0),
            "error": result.get("error", None),
            "prompt": prompt,
        }
        outputs.append((tool_returned_str, tool_returned_images, tool_stat))

    return outputs

async def _async_generate_image(
    prompt: str,
    base_url: str = _DEFAULT_BASE_URL,
    size: str = _DEFAULT_SIZE,
    num_inference_steps: int = _DEFAULT_STEPS,
    guidance_scale: float = _DEFAULT_GUIDANCE,
    seed: int = None,
    timeout: int = _DEFAULT_TIMEOUT,
) -> Dict:
    api_endpoint = f"{base_url}/v1/images/generations"
    payload = {
        "prompt": prompt,
        "size": size,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
    }
    if seed is not None:
        payload["seed"] = seed

    start_time = time.time()

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                api_endpoint,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as response:

                duration = time.time() - start_time

                if response.status != 200:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "image": None,
                        "error": f"HTTP {response.status}: {error_text}",
                        "duration": duration,
                    }

                result = await response.json()

                pil_image = await _decode_image_from_response(
                    session, api_endpoint, result, timeout
                )

                return {
                    "success": True,
                    "image": pil_image,
                    "error": None,
                    "duration": duration,
                }

    except asyncio.TimeoutError:
        return {
            "success": False,
            "image": None,
            "error": f"Timeout after {timeout}s",
            "duration": time.time() - start_time,
        }
    except Exception as e:
        return {
            "success": False,
            "image": None,
            "error": str(e),
            "duration": time.time() - start_time,
        }

def call_image_generate(
    prompt: str,
    base_url: str = _DEFAULT_BASE_URL,
    size: str = _DEFAULT_SIZE,
    num_inference_steps: int = _DEFAULT_STEPS,
    guidance_scale: float = _DEFAULT_GUIDANCE,
    seed: int = None,
    timeout: int = _DEFAULT_TIMEOUT,
) -> Tuple[str, List[Image.Image], Dict]:
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    asyncio.run,
                    _async_generate_image(
                        prompt=prompt,
                        base_url=base_url,
                        size=size,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        seed=seed,
                        timeout=timeout,
                    ),
                )
                result = future.result()
        else:
            result = loop.run_until_complete(
                _async_generate_image(
                    prompt=prompt,
                    base_url=base_url,
                    size=size,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                    timeout=timeout,
                )
            )
    except RuntimeError:
        result = asyncio.run(
            _async_generate_image(
                prompt=prompt,
                base_url=base_url,
                size=size,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                timeout=timeout,
            )
        )

    tool_returned_images: List[Image.Image] = []

    if result["success"]:
        pil_image = result["image"]
        tool_returned_images.append(pil_image)

        tool_returned_str = (
            "[Image Generation Results] "
            "The image has been successfully generated based on your prompt. "
            "Here is the generated image:\n"
            "<|vision_start|><|image_pad|><|vision_end|>\n"
            f"prompt used: {prompt}\n"
        )
    else:
        error_msg = result.get("error", "Unknown error")
        print(f"[call_image_generate] Generation failed: {error_msg}")

        tool_returned_str = (
            "[Image Generation Results] "
            "There is an error encountered in generating the image. "
            "Please refine your prompt or reason with your own capabilities.\n"
            f"Error: {error_msg}\n"
        )

    tool_stat = {
        "success": result["success"],
        "num_images": len(tool_returned_images),
        "duration": result.get("duration", 0.0),
        "error": result.get("error", None),
        "prompt": prompt,
    }

    return tool_returned_str, tool_returned_images, tool_stat
