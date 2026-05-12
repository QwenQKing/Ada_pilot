# glm_ocr_server.py
import asyncio
import base64
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Optional
import time
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from modelscope import AutoModelForImageTextToText, AutoProcessor
from pydantic import BaseModel

# ========== 配置 ==========
MODEL_PATH = os.environ.get("GLM_OCR_MODEL_PATH", "models/GLM-OCR")
MAX_QUEUE_SIZE = 100       # 最大排队数
MAX_NEW_TOKENS = 8192
PORT = int(os.environ.get("GLM_OCR_SERVER_PORT", 4512))

# ========== 全局变量 ==========
processor = None
model = None
request_queue: asyncio.Queue = None
worker_task = None


# ========== 模型加载 & 后台worker ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    global processor, model, request_queue, worker_task

    print("Step 1: 开始加载 processor...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    print("Step 2: Processor 加载完成，开始加载 model...", flush=True)
    
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="cuda:0",
    )
    print("Step 3: Model 加载完成！", flush=True)
    model.eval()
    print(f"Step 4: 模型设备: {model.device}", flush=True)

    request_queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
    worker_task = asyncio.create_task(inference_worker())
    print("Step 5: 服务初始化完成！", flush=True)

    yield

    worker_task.cancel()
    print("Server shutting down.")


async def inference_worker():
    """后台串行处理队列中的推理请求"""
    while True:
        future, image_path, prompt = await request_queue.get()
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, run_inference, image_path, prompt
            )
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
        finally:
            request_queue.task_done()


def run_inference(image_path: str, prompt: str) -> str:
    t0 = time.time()
    messages = [{"role": "user", "content": [
        {"type": "image", "url": image_path},
        {"type": "text",  "text": prompt}
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device)
    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

    result = processor.decode(
        generated_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    print(f"[glm_ocr_server] inference time: {time.time()-t0:.2f}s", flush=True)
    return result


# ========== FastAPI ==========
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class OCRRequest(BaseModel):
    image_path: Optional[str] = None       # 服务器本地路径
    image_base64: Optional[str] = None     # base64编码图片（远程调用用这个）
    prompt: str = "Text Recognition:"


@app.post("/ocr")
async def ocr(req: OCRRequest):
    if request_queue.full():
        raise HTTPException(status_code=429, detail="Server busy, queue is full")

    # 处理图片来源
    tmp_file = None
    try:
        if req.image_base64:
            # 远程传base64图片，先存成临时文件
            img_data = base64.b64decode(req.image_base64)
            tmp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp_file.write(img_data)
            tmp_file.close()
            image_path = tmp_file.name

        elif req.image_path:
            if not os.path.exists(req.image_path):
                raise HTTPException(status_code=400, detail="Image file not found")
            image_path = req.image_path
        else:
            raise HTTPException(status_code=400, detail="Need image_path or image_base64")

        # 放入队列等待推理
        future = asyncio.get_event_loop().create_future()
        await request_queue.put((future, image_path, req.prompt))
        result = await future

        return {"result": result, "queue_size": request_queue.qsize()}

    finally:
        if tmp_file and os.path.exists(tmp_file.name):
            os.unlink(tmp_file.name)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "queue_size": request_queue.qsize(),
        "max_queue_size": MAX_QUEUE_SIZE
    }

class ChatCompletionRequest(BaseModel):
    model: str = "glm-ocr"
    messages: list
    max_tokens: int = 8192


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    # 队列超过一半就拒绝，让客户端快速失败后重试，而不是等到超时
    if request_queue.qsize() > MAX_QUEUE_SIZE // 2:
        raise HTTPException(
            status_code=503,
            detail=f"Server busy, queue={request_queue.qsize()}/{MAX_QUEUE_SIZE}, retry later"
        )
    image_b64 = None
    prompt = "Text Recognition:"

    for msg in req.messages:
        if msg.get("role") == "user":
            content = msg.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "image_url":
                        url = item["image_url"]["url"]
                        # 支持 data:image/png;base64,xxx 格式
                        if "," in url:
                            image_b64 = url.split(",")[1]
                        else:
                            image_b64 = url
                    elif item.get("type") == "text":
                        prompt = item["text"]
            elif isinstance(content, str):
                prompt = content

    if not image_b64:
        raise HTTPException(status_code=400, detail="No image found in messages")

    ocr_req = OCRRequest(image_base64=image_b64, prompt=prompt)
    result = await ocr(ocr_req)

    return {
        "id": "chatcmpl-glmocr",
        "object": "chat.completion",
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": result["result"]
            },
            "finish_reason": "stop"
        }]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, workers=1)  # workers必须=1，GPU不能多进程