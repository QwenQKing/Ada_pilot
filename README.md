# AdaPilot: Towards Scene-Adaptive Policy Learning for Cross-Generator Text-to-Image Quality Optimization

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-NeurIPS%202026-b31b1b.svg)](https://anonymous.4open.science/r/AdaPilot-0A37)
[![Code](https://img.shields.io/badge/Code-Anonymous-black.svg)](https://anonymous.4open.science/r/AdaPilot-0A37)

### AdaPilot: Scene-Adaptive Policy Learning for Cross-Generator Text-to-Image Quality Optimization

[📄 Paper](https://anonymous.4open.science/r/AdaPilot-0A37) | [🚀 Quick Start](#quick-start)

</div>

---

## Overview

<div align="center">
  <img src="figs/fig-1.png" width="85%"/>
</div>

**AdaPilot** is a scene-adaptive policy learning framework for cross-generator text-to-image quality optimization. It trains a multimodal large language model (MLLM) agent via end-to-end reinforcement learning to iteratively improve image generation quality through multi-turn visual feedback — with the image generator kept **fully frozen**.

Unlike prior methods that are tightly coupled with a specific generator, AdaPilot decouples the policy from generator internals and learns to **zero-shot transfer** across unseen generators without any retraining.

<div align="center">
  <img src="figs/AdaPilot.png" width="95%"/>
</div>

**Three core mechanisms:**

- **Adaptive Multi-Turn Generation.** The agent (Qwen2.5-VL-7B) iteratively reasons, generates optimized prompts, assesses the returned image, and decides whether to refine or terminate — forming a closed-loop visual feedback MDP.
- **AdaReward (Scene-Adaptive Quality Reward).** Different scene types (poster, e-commerce, text-rich, etc.) activate different metric subsets (CLIP-T, HPS, Aesthetic, GDino, OCR) with renormalized weights, plus a process-level monotonic improvement constraint across turns.
- **Cross-Generator Policy Decoupling.** The policy conditions only on visual observations, not generator internals, enabling zero-shot transfer to unseen generators (FLUX.2, LongCat, Ovis, Z-Image, GPT-Image-1, Gemini) at inference time.

---

## Experimental Results

**Qualitative comparison with baselines on a sample prompt:**

<div align="center">
  <img src="figs/case_study.png" width="90%"/>
</div>

**Cross-generator transfer — OOD datasets (4 unseen generators, radar charts):**

<div align="center">
  <img src="figs/ood_radar_4.png" width="90%"/>
</div>

**Monotonic quality improvement across interaction rounds on 10 benchmarks:**

<div align="center">
  <img src="figs/rounds.png" width="90%"/>
</div>

AdaPilot outperforms all baselines on **6 in-distribution** and **4 out-of-distribution** datasets, and a single policy zero-shot drives unseen generators while maintaining quality gains — with largest improvements on weaker generators.

---

## Quick Start

### Environment Setup

```bash
conda create -n adapilot python==3.12 -y
conda activate adapilot
```

Install verl from source:

```bash
cd adapilot/verl
pip3 install -e ".[gpu,vllm]"
cd ../..
```

Install flash-attn (pre-built wheel required — standard pip build often fails):

```bash
pip3 install flash-attn==2.7.4.post1
# If the above fails, download the matching wheel from:
# https://github.com/Dao-AILab/flash-attention/releases
```

Install remaining dependencies:

```bash
pip3 install -r adapilot_pip_freeze.txt
```

### Reward Model Deployment

AdaPilot's AdaReward relies on four self-hosted scoring servers. All servers expose a FastAPI HTTP interface and support dynamic batching. Deploy them before launching training or evaluation.

#### Model Downloads

| Server | Model | Source |
|---|---|---|
| CLIP | `openai/clip-vit-large-patch14` | [HuggingFace](https://huggingface.co/openai/clip-vit-large-patch14) |
| HPS v2.1 | `ViT-H-14` + `HPS_v2.1_compressed.pt` | [HuggingFace](https://huggingface.co/xswu/HPSv2) |
| GroundingDINO | `IDEA-Research/grounding-dino-base` | [HuggingFace](https://huggingface.co/IDEA-Research/grounding-dino-base) |
| GLM-OCR | `ZhipuAI/GLM-OCR` | [ModelScope](https://modelscope.cn/models/ZhipuAI/GLM-OCR) |

Download models to a local directory (e.g. `models/`) and set the paths via environment variables shown below.

#### CLIP Server — port 8030

Computes CLIP-T (text-image alignment) and CLIP-I (image-image similarity).

```bash
export CUDA_VISIBLE_DEVICES=7
export CLIP_MODEL_PATH=models/clip-vit-large-patch14   # local HF model dir
export CLIP_SERVER_PORT=8030
export CLIP_MAX_BATCH_SIZE=16
export CLIP_BATCH_TIMEOUT=0.05

cd rewardmodel && python clip_server.py
# or use the convenience script:
bash rewardmodel/start_clip_server.sh
```

Endpoints: `POST /clip_t` (image + caption → score), `POST /clip_i` (image\_gen + image\_gt → score), `GET /health`.

#### HPS v2.1 Server — port 8020

Computes human preference scores using HPSv2.1 (ViT-H-14).

```bash
export CUDA_VISIBLE_DEVICES=6
export HPS_MODEL_DIR=models/HPSv2                      # dir containing HPS_v2.1_compressed.pt
export HPS_SERVER_PORT=8020
export HPS_MAX_BATCH_SIZE=16
export HPS_BATCH_TIMEOUT=0.05

cd rewardmodel && python hps_server.py
# or:
bash rewardmodel/start_hps.sh
```

Endpoint: `POST /score` (image + prompt → raw\_score, normalized\_score), `GET /health`.

#### GroundingDINO Server — port 8010

Runs zero-shot object detection for entity grounding reward.

```bash
export CUDA_VISIBLE_DEVICES=5
export GDINO_MODEL_PATH=models/GroundingDINO           # local HF model dir

cd rewardmodel && uvicorn gdino_server:app --host 0.0.0.0 --port 8010
# or:
bash rewardmodel/start_gdino.sh
```

Endpoint: `POST /detect` (image + queries → boxes, scores, labels), `GET /health`.

#### GLM-OCR Server — port 4512

Provides OCR-based text rendering accuracy reward via GLM-OCR. Also exposes an OpenAI-compatible `/v1/chat/completions` endpoint so it can be called through `GLM_OCR_API_BASE`.

```bash
export CUDA_VISIBLE_DEVICES=4
export GLM_OCR_MODEL_PATH=models/GLM-OCR              # local ModelScope model dir
export GLM_OCR_SERVER_PORT=4512

bash rewardmodel/glm-ocr.sh
# or equivalently:
CUDA_DEVICE=4 PORT=4512 bash rewardmodel/glm-ocr.sh
```

Endpoint: `POST /ocr` (image\_base64 + prompt → result), `POST /v1/chat/completions` (OpenAI-compatible), `GET /health`.

#### Environment Variables for Training / Evaluation

Once all servers are running, point AdaPilot to them:

| Variable | Description |
|---|---|
| `IMAGE_GEN_BASE_URL` | Image generation API endpoint (OpenAI-compatible) |
| `CLIP_SERVER_URL` | e.g. `http://localhost:8030` |
| `HPS_SERVER_URL` | e.g. `http://localhost:8020` |
| `GDINO_SERVER_URL` | e.g. `http://localhost:8010` |
| `GLM_OCR_API_BASE` | e.g. `http://localhost:4512` |
| `VLM_API_BASE` / `VLM_MODEL` | VLM judge for aesthetic scoring and entity extraction — set to an OpenAI-compatible endpoint serving **Qwen2.5-VL-72B-Instruct** (used during training only) |

> **Training-time VLM judge.** Aesthetic quality assessment and entity extraction in AdaReward call an external VLM API. During our experiments we used `Qwen2.5-VL-72B-Instruct` served via a vLLM OpenAI-compatible endpoint. Set `VLM_MODEL=Qwen2.5-VL-72B-Instruct` and point `VLM_API_BASE` to your server. This dependency is only required for training (reward computation); inference does not need it.

Servers can also be replaced by passing the local model path instead of a URL (e.g. `CLIP_MODEL_PATH` instead of `CLIP_SERVER_URL`), in which case the reward code loads the model in-process.

### Training

```bash
export BASE_MODEL=/path/to/Qwen2.5-VL-7B-Instruct
export IMAGE_GEN_BASE_URL=http://your-image-gen-server
# ... set other service URLs above

bash adapilot/scripts/train_adapilot.sh
```

The script starts a Ray cluster, then runs GRPO training via `adapilot.trainer.multimodal.main_ppo`.

**Key hyperparameters** (configurable in `train_adapilot.sh`):

| Parameter | Default | Description |
|---|---|---|
| `data.train_batch_size` | 32 | Global batch size |
| `data.max_prompt_length` | 8192 | Max prompt tokens |
| `actor_rollout_ref.rollout.n` | 4 | Rollout samples per prompt |
| `actor_rollout_ref.rollout.max_gen_round` | 6 | Max multi-turn rounds |
| `actor_rollout_ref.actor.optim.lr` | 2e-6 | Learning rate |
| `trainer.n_gpus_per_node` | 4 | GPUs per node |
| `trainer.total_epochs` | 2 | Training epochs |

Our experiments used 2 servers × 8 NVIDIA H100 (80GB) GPUs.

### Inference

Merge FSDP checkpoint shards and run inference:

```bash
export BASE_MODEL=/path/to/Qwen2.5-VL-7B-Instruct
export CKPT_DIR=/path/to/checkpoints/adapilot/adapilot_train
export GLOBAL_STEP=120
export IMAGE_GEN_BASE_URL=http://your-image-gen-server

bash adapilot/scripts/inference_torch.sh
```

Results are saved to `results/adapilot_results/`.

To use a vLLM-accelerated inference server instead:

```bash
bash adapilot/scripts/inference_vllm.sh
```

### Evaluation

```bash
python adapilot/eval_metrics.py --help
```

Evaluation covers the same reward dimensions as training: CLIP-T, HPS, Aesthetic (VLM judge), GroundingDINO, and OCR accuracy.

---

## BibTeX

If you find this work helpful, please cite:

```bibtex
@inproceedings{adapilot2026,
  title     = {AdaPilot: Towards Scene-Adaptive Policy Learning for Cross-Generator Text-to-Image Quality Optimization},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
}
```
