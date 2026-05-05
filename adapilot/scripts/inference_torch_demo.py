import argparse
import hashlib
import json
import os
import pickle
import re

import pandas as pd
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from transformers import AutoConfig
from adapilot.utils.tools.image_generate import call_image_generate

MAX_PIXELS = 672 * 672
MIN_ROUNDS = 4

FORCE_FINISH_USER_TEXT = (
    "Since this is the final round, you must output <finish>over</finish> "
    "to complete the task and must not output <paint>.\n"
    "Please output in the following format:\n"
    "<reasoning>your reasoning process here</reasoning>\n"
    "<assess>your visual assessment here</assess>\n"
    "<finish>over</finish>"
)

def _build_synth_last_text(decoded_resp: str) -> str:
    reasoning_m = re.search(r'<reasoning>(.*?)</reasoning>', decoded_resp, re.DOTALL)
    reasoning_raw = reasoning_m.group(1).strip() if reasoning_m else ""
    reasoning_txt = re.sub(r'<[^>]+>', '', reasoning_raw).strip() or "Reached maximum paint rounds."

    text_without_reasoning = re.sub(r'<reasoning>.*?</reasoning>', '', decoded_resp, flags=re.DOTALL)
    assess_m = re.search(r'<assess>(.*?)</assess>', text_without_reasoning, re.DOTALL)
    assess_raw = assess_m.group(1).strip() if assess_m else ""
    assess_txt = re.sub(r'<[^>]+>', '', assess_raw).strip() or "Submitting current best image."

    return (
        f"<reasoning>{reasoning_txt}</reasoning>"
        f"<assess>{assess_txt}</assess>"
        f"<finish>over</finish>"
    )

def _check_round_format(assistant_block: str, kind: str) -> tuple:
    r = assistant_block.strip()
    spec = {
        'first':  {'reasoning': 1, 'paint': 1, 'assess': 0, 'finish': 0},
        'middle': {'reasoning': 1, 'paint': 1, 'assess': 1, 'finish': 0},
        'last':   {'reasoning': 1, 'paint': 0, 'assess': 1, 'finish': 1},
    }[kind]
    for tag, expect in spec.items():
        got_open  = r.count(f'<{tag}>')
        got_close = r.count(f'</{tag}>')
        if got_open != expect or got_close != expect:
            return False, f"count(<{tag}>)={got_open} count(</{tag}>)={got_close} expect {expect}"
    def _inner(tag):
        m = re.search(f'<{tag}>(.*?)</{tag}>', r, re.DOTALL)
        return m.group(1).strip() if m else ''
    if kind == 'first' and not (_inner('reasoning') and _inner('paint')):
        return False, "empty reasoning or paint"
    if kind == 'middle' and not (_inner('reasoning') and _inner('assess') and _inner('paint')):
        return False, "empty reasoning/assess/paint"
    if kind == 'last':
        if not (_inner('reasoning') and _inner('assess')):
            return False, "empty reasoning or assess"
        if _inner('finish') != 'over':
            return False, f"finish!='over' (got '{_inner('finish')}')"
    return True, ""

def parse_args():
    parser = argparse.ArgumentParser()

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--caption', type=str,
                             help='Single caption string')
    input_group.add_argument('--parquet', type=str,
                             help='Parquet file for batch inference')
    input_group.add_argument('--data_root', type=str,

    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--base_model_path', type=str, default=None,
                    help='Base model path for loading tokenizer/config (needed for verl checkpoints)')

    parser.add_argument('--prompt_path', type=str,
                        default='adapilot/prompts/round_1_user_prompt_qwenvl.pkl')
    parser.add_argument('--after_generate_prompt_path', type=str,
                        default='adapilot/prompts/user_prompt_after_image_generate.pkl')

    parser.add_argument('--max_gen_round', type=int, default=6,
                        help='Aligned with actor_rollout_ref.rollout.max_gen_round')
    parser.add_argument('--max_img_draw_round', type=int, default=4,
                        help='Aligned with actor_rollout_ref.rollout.generate.image_generate_limit')
    parser.add_argument('--max_new_tokens', type=int, default=1024)

    parser.add_argument('--sample_size', type=int, default=None,
                        help='Only run first N samples (for quick test)')

    parser.add_argument('--image_save_dir', type=str,
                        default='adapilot/data/PosterCraft/generated_images',
)
    parser.add_argument('--output_json', type=str,
)
    parser.add_argument('--result_root', type=str,
    parser.add_argument('--output_basename', type=str, default='val_raw.json',
    return parser.parse_args()

def load_model_and_processor(model_path, base_model_path=None):
    config_path = base_model_path if base_model_path else model_path
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        config=AutoConfig.from_pretrained(config_path),
        torch_dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(config_path)
    return model, processor

def generate_response(model, processor, messages, max_new_tokens=1024):
    sanitized = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            new_content = []
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text":
                    txt = c.get("text", "")
                    txt = txt.replace("<image>", "").replace("<|image_pad|>", "")
                    new_c = dict(c)
                    new_c["text"] = txt
                    new_content.append(new_c)
                else:
                    new_content.append(c)
            sanitized.append({**msg, "content": new_content})
        else:
            sanitized.append(msg)

    text = processor.apply_chat_template(
        sanitized, tokenize=False, add_generation_prompt=True
    )

    image_inputs = []
    for msg in sanitized:
        for content in msg.get("content", []):
            if isinstance(content, dict) and content.get("type") == "image":
                image_inputs.append(content["image"])

    inputs = processor(
        text=[text],
        images=image_inputs if image_inputs else None,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    generated_ids = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False
    )
    trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

def run_single(
    caption, data_source, ground_truth,
    model, processor,
    round_1_prompt, user_prompt_after_generate,
    max_gen_round, max_img_draw_round, max_new_tokens,
    image_save_dir, sample_id,
):
    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": f"{round_1_prompt}\n{caption}"}],
    }]

    input_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    img_draw_cnt = 0
    output_texts = []
    rounds = []
    final_image_url = None
    format_violations = []
    force_finish_injected = False

    print(f"\n{'='*60}")
    print(f"[Sample {sample_id}] {caption[:80]}...")

    for current_round in range(max_gen_round):
        print(f"[Round #{current_round}] Generating ...")
        response_text = generate_response(model, processor, messages, max_new_tokens)
        print(f"[Round #{current_round}] Response: {response_text[:120]}...")

        output_texts.append(response_text)
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": response_text}],
        })

        _has_finish_this_round = bool(re.search(r'<finish>.*?</finish>', response_text, re.DOTALL))
        if _has_finish_this_round:
            _kind = 'last'
        elif current_round == 0:
            _kind = 'first'
        else:
            _kind = 'middle'
        _ok, _reason = _check_round_format(response_text, _kind)
        if not _ok:
            format_violations.append((current_round, _kind, _reason))
            print(f"[FORMAT WARN] Round #{current_round} kind={_kind} violates: {_reason}")

        if _has_finish_this_round:
            rounds.append({
                "round": current_round,
                "assistant_text": response_text,
                "image_url": None,
            })
            print(f"[Round #{current_round}] <finish> detected. Done.")
            break

        paint_match = re.search(r'<paint>(.*?)</paint>', response_text, re.DOTALL)

        if not paint_match:
            rounds.append({
                "round": current_round,
                "assistant_text": response_text,
                "image_url": None,
            })
            print(f"[Round #{current_round}] No <paint> and no <finish>. Done.")
            break

        is_last_iter = (current_round == max_gen_round - 1)
        at_paint_limit = (img_draw_cnt >= max_img_draw_round)
        if at_paint_limit or is_last_iter:
            need_synth_last = is_last_iter or force_finish_injected

            if not need_synth_last:
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": FORCE_FINISH_USER_TEXT}],
                })
                force_finish_injected = True
                rounds.append({
                    "round": current_round,
                    "assistant_text": response_text,
                    "image_url": None,
                    "note": "paint at limit, injected force-finish, retry next round",
                })
                print(
                    f"[Round #{current_round}] paint limit (cnt={img_draw_cnt}>="
                    f"{max_img_draw_round}); injected force-finish, will retry"
                )
                continue

            synth_text = _build_synth_last_text(response_text)
            output_texts.append(synth_text)
            if messages and messages[-1].get("role") == "assistant":
                messages[-1] = {
                    "role": "assistant",
                    "content": [{"type": "text", "text": synth_text}],
                }
            reason = (
                "STILL output <paint> after force-finish" if force_finish_injected
                else "reached last iter, no next generate chance"
            )
            rounds.append({
                "round": current_round,
                "assistant_text": synth_text,
                "image_url": None,
                "synthesized": True,
                "note": f"synth-last: {reason}",
            })
            print(
                f"[Round #{current_round}] paint limit, synthesized last "
                f"({reason}): {synth_text[:80]}..."
            )
            break

        paint_prompt = None
        for m in re.finditer(r'<paint>(.*?)</paint>', response_text, re.DOTALL):
            paint_prompt = m.group(1).strip()

        if not paint_prompt:
            rounds.append({
                "round": current_round,
                "assistant_text": response_text,
                "image_url": None,
            })
            print(f"[Round #{current_round}] Empty <paint> tag. Done.")
            break

        print(f"[Round #{current_round}] Calling image generate: '{paint_prompt[:60]}' ...")
        tool_returned_str, tool_returned_images, tool_stat = call_image_generate(
            prompt=paint_prompt
        )
        img_draw_cnt += 1
        print(f"[Round #{current_round}] Generate {'SUCCESS' if tool_stat['success'] else 'FAILED'}")

        image_save_path = None
        if tool_stat["success"] and tool_returned_images:
            prompt_hash = hashlib.md5(tool_stat["prompt"].encode()).hexdigest()[:8]
            image_save_path = os.path.join(
                image_save_dir,
                f"sample{sample_id:04d}_round{current_round:02d}_{prompt_hash}.png"
            )
            tool_returned_images[0].save(image_save_path)
            final_image_url = image_save_path
            print(f"[Round #{current_round}] Saved -> {image_save_path}")

        rounds.append({
            "round": current_round,
            "assistant_text": response_text,
            "image_url": image_save_path,
        })

        if (
            "[Image Generation Results]" in tool_returned_str
            and "There is an error" not in tool_returned_str
            and user_prompt_after_generate is not None
        ):
            next_text = tool_returned_str + user_prompt_after_generate
        else:
            next_text = tool_returned_str

        if tool_stat["success"] and tool_returned_images:
            next_user_content = [
                {"type": "text", "text": next_text},
                {
                    "type": "image",
                    "image": tool_returned_images[0],
                    "max_pixels": MAX_PIXELS,
                },
            ]
        else:
            next_user_content = [{"type": "text", "text": next_text}]

        messages.append({"role": "user", "content": next_user_content})

    else:
        print(f"[Sample {sample_id}] Reached max_gen_round={max_gen_round}.")

    n_rounds = len(output_texts)
    format_ok = (len(format_violations) == 0)
    meets_min_rounds = (n_rounds >= MIN_ROUNDS)
    would_get_positive_reward = format_ok and meets_min_rounds

    if not format_ok:
        print(f"[Sample {sample_id}] FORMAT VIOLATIONS: {len(format_violations)} round(s)")
        for rnd, k, rsn in format_violations:
            print(f"  - round {rnd} ({k}): {rsn}")
    if not meets_min_rounds:
        print(
            f"[Sample {sample_id}] ROUND COUNT WARN: n={n_rounds} < MIN_ROUNDS={MIN_ROUNDS} "
            f"→ reward would be capped at { (n_rounds-1)/MIN_ROUNDS :.2f} in training"
        )
    print(
        f"[Sample {sample_id}] FORMAT SUMMARY: "
        f"n_rounds={n_rounds} format_ok={format_ok} meets_min_rounds={meets_min_rounds} "
        f"would_pass_format_gate={would_get_positive_reward}"
    )

    gt_list = (
        ground_truth if isinstance(ground_truth, list)
        else ([ground_truth] if ground_truth else [])
    )

    record = {
        "data_source":     data_source,
        "input_text":      input_text,
        "output_texts":    output_texts,
        "caption":         caption,
        "ground_truth":    gt_list,
        "image_url":       final_image_url,
        "final_image_url": final_image_url,
        "format_check": {
            "n_rounds":              n_rounds,
            "violations":            [
                {"round": r, "kind": k, "reason": rsn}
                for r, k, rsn in format_violations
            ],
            "format_ok":             format_ok,
            "meets_min_rounds":      meets_min_rounds,
            "would_pass_format_gate": would_get_positive_reward,
        },
        "rounds":          rounds,
    }
    return record

def build_samples_from_parquet(parquet_path: str, sample_size: int | None = None) -> list:
    df = pd.read_parquet(parquet_path)
    if sample_size is not None:
        df = df.iloc[:sample_size]
        print(f"[Info] sample_size={sample_size}, truncated to {len(df)} rows")

    samples = []
    for idx, row in df.iterrows():
        prompt_field = row.get('prompt', None)
        caption = ""
        if prompt_field is not None:
            try:
                if hasattr(prompt_field, '__len__') and len(prompt_field) > 0:
                    first = prompt_field[0]
                    if isinstance(first, dict):
                        caption = first.get('content', '') or first.get('text', '')
                    else:
                        caption = str(first)
                else:
                    caption = str(prompt_field)
            except (IndexError, TypeError, KeyError):
                caption = str(prompt_field)
        reward_model = row.get('reward_model', {})
        ground_truth = (
            reward_model.get('ground_truth', None)
            if isinstance(reward_model, dict) else None
        )
        extra_info = row.get('extra_info', {})
        sample_id = (
            int(extra_info.get('index', idx))
            if isinstance(extra_info, dict) else idx
        )
        samples.append({
            "caption":      caption,
            "data_source":  row.get('data_source', 'unknown'),
            "ground_truth": ground_truth,
            "sample_id":    sample_id,
        })
    return samples

def discover_parquets(data_root: str) -> list:
    import glob
    found = []
    for sub in sorted(glob.glob(os.path.join(data_root, "*"))):
        if not os.path.isdir(sub):
            continue
        name = os.path.basename(sub)
        if name in ("mixed", "datatxt"):
            print(f"[Skip ] {name}")
            continue
        parquet = None
        for pat in ("*_test.parquet", "*_test.pq"):
            matches = sorted(glob.glob(os.path.join(sub, pat)))
            if matches:
                parquet = matches[0]
                break
        if parquet:
            found.append((name, parquet))
            print(f"[Found] {name} <- {parquet}")
        else:
            print(f"[Warn ] {name}: no *_test.parquet, skip")
    return found

def infer_samples(samples, output_json, image_save_dir,
                  model, processor, round_1_prompt, user_prompt_after_generate, args):
    os.makedirs(image_save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_json)) or ".", exist_ok=True)

    print(f"[Info] {len(samples)} samples → {output_json}")
    all_results = []
    for sample in samples:
        record = run_single(
            caption=sample['caption'],
            data_source=sample['data_source'],
            ground_truth=sample['ground_truth'],
            model=model,
            processor=processor,
            round_1_prompt=round_1_prompt,
            user_prompt_after_generate=user_prompt_after_generate,
            max_gen_round=args.max_gen_round,
            max_img_draw_round=args.max_img_draw_round,
            max_new_tokens=args.max_new_tokens,
            image_save_dir=image_save_dir,
            sample_id=sample['sample_id'],
        )
        all_results.append(record)
        print(f"  final_image_url : {record['final_image_url']}")
        print(f"  ground_truth    : {record['ground_truth']}")

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {len(all_results)} records -> {output_json}\n")

def main():
    args = parse_args()

    print(f"[Loading] {args.model_path} ...")
    model, processor = load_model_and_processor(
        args.model_path,
        base_model_path=args.base_model_path,
    )

    with open(args.prompt_path, 'rb') as f:
        round_1_prompt = pickle.load(f)
    round_1_prompt = round_1_prompt.replace("<image>", "").strip()

    user_prompt_after_generate = None
    if args.after_generate_prompt_path and os.path.exists(args.after_generate_prompt_path):
        with open(args.after_generate_prompt_path, 'rb') as f:
            user_prompt_after_generate = pickle.load(f)
    print(f"[Prompt] after_generate={'loaded' if user_prompt_after_generate else 'None'}")

    common = dict(
        model=model, processor=processor,
        round_1_prompt=round_1_prompt,
        user_prompt_after_generate=user_prompt_after_generate,
        args=args,
    )

    if args.data_root:
        if not args.result_root:
            raise SystemExit("[error] --data_root needs --result_root")
        datasets = discover_parquets(args.data_root)
        if not datasets:
            raise SystemExit(f"[error] no *_test.parquet under {args.data_root}")
        print(f"[Info] batch mode: {len(datasets)} dataset(s)\n")
        for name, parquet in datasets:
            save_dir = os.path.join(args.result_root, name)
            print(f"=== [{name}] {parquet} ===")
            infer_samples(
                samples=build_samples_from_parquet(parquet, args.sample_size),
                output_json=os.path.join(save_dir, args.output_basename),
                image_save_dir=os.path.join(save_dir, "generated_images"),
                **common,
            )

    elif args.parquet:
        if not args.output_json:
            raise SystemExit("[error] --parquet needs --output_json")
        infer_samples(
            samples=build_samples_from_parquet(args.parquet, args.sample_size),
            output_json=args.output_json,
            image_save_dir=args.image_save_dir,
            **common,
        )

    else:
        if not args.output_json:
            raise SystemExit("[error] --caption needs --output_json")
        infer_samples(
            samples=[{
                "caption":      args.caption,
                "data_source":  "unknown",
                "ground_truth": None,
                "sample_id":    0,
            }],
            output_json=args.output_json,
            image_save_dir=args.image_save_dir,
            **common,
        )

if __name__ == "__main__":
    main()