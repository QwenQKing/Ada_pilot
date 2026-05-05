import os
import copy
import pickle
import re
import threading
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from copy import deepcopy
from typing import Any, List, Union
import hashlib
import numpy as np
import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from tqdm import tqdm
from verl import DataProto
from verl.third_party.vllm import vllm_version
from verl.utils import hf_processor
from verl.utils.torch_functional import pad_sequence_to_length
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import (
    _repeat_interleave,
    _pre_process_inputs,  
    vLLMRollout,
)

from adapilot.utils.tools.image_generate import call_image_generate, call_image_generate_batch
from adapilot.utils.torch_functional import get_final_eos_mask

def pad_to_max_stack(tensor_list: List[torch.Tensor], pad_token_id: int, dim: int) -> torch.Tensor:
    assert all([t.ndim == 1 for t in tensor_list])
    max_len = max([t.size(0) for t in tensor_list])
    padded_tensor_list = []
    for t in tensor_list:
        padded_tensor_list.append(
            torch.cat([t, torch.tensor([pad_token_id] * (max_len - t.size(0)), device=t.device, dtype=t.dtype)], dim=0)
        )
    return torch.stack(padded_tensor_list, dim=dim)

class vLLMRollout_MultiTurn_AdaPilot(vLLMRollout):

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):

        super().__init__(model_path, config, tokenizer, model_hf_config, **kwargs)

        self.tokenizer = tokenizer
        self.processor = hf_processor(model_path)

        self.user_prompt_after_image_generate = None
        try:
            with open(config.generate.user_prompt_after_image_generate, 'rb') as file:
                self.user_prompt_after_image_generate = pickle.load(file)
        except Exception as e:
            print(f"Error: {e} | user_prompt_after_image_generate default to None")

        print(f"[Prompt Set] user_prompt_after_image_generate: {self.user_prompt_after_image_generate}")

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:

        import debugpy
    
        rank = int(os.environ.get("RANK", 0))
        
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids']
        batch_size = idx.size(0)
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']
        eos_token_id = prompts.meta_info['eos_token_id']
        input_prompt_generation_mask = torch.zeros_like(
            idx, dtype=attention_mask.dtype, device=attention_mask.device
        )

        non_tensor_batch = prompts.non_tensor_batch
        if 'raw_prompt_ids' not in non_tensor_batch:
            non_tensor_batch['raw_prompt_ids'] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object
            )

        if batch_size != len(non_tensor_batch['raw_prompt_ids']):
            raise RuntimeError('vllm sharding manager is not work properly.')

        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1,
            }

        n = 1 if prompts.meta_info.get('validate', False) else self.config.n

        vllm_inputs = []
        multi_turn_response_mask = []
        prefix_prompt_lengths = []
        generate_tool_return_images = []

        if 'multi_modal_data' in non_tensor_batch:
            _multi_modal_data_list = non_tensor_batch['multi_modal_data']
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop('raw_prompt_ids'), _multi_modal_data_list):
                prefix_length = len(raw_prompt_ids)
                for _ in range(n):
                    vllm_inputs.append({
                        'prompt_token_ids': deepcopy(raw_prompt_ids),
                        'multi_modal_data': deepcopy(multi_modal_data),
                    })
                    multi_turn_response_mask.append([
                        torch.zeros(prefix_length, dtype=attention_mask.dtype, device=attention_mask.device)
                    ])
                    prefix_prompt_lengths.append(prefix_length)
                    generate_tool_return_images.append([])

        else:
            for raw_prompt_ids in non_tensor_batch.pop('raw_prompt_ids'):
                prefix_length = len(raw_prompt_ids)
                for _ in range(n):
                    vllm_inputs.append({
                        'prompt_token_ids': deepcopy(raw_prompt_ids),
                    })
                    multi_turn_response_mask.append([
                        torch.zeros(prefix_length, dtype=attention_mask.dtype, device=attention_mask.device)
                    ])
                    prefix_prompt_lengths.append(prefix_length)
                    generate_tool_return_images.append([])

        to_generate = list(range(batch_size * n))
        worker_trajs_count = len(to_generate)
        id_img_draw_cnt = [0] * (batch_size * n)
        id_force_finish_injected = [False] * (batch_size * n)
        max_img_draw_round = self.config.generate.image_generate_limit
        save_dir = getattr(self.config.generate, "image_save_dir", "/tmp/paint_images")

        img_gen_total_calls = 0
        img_gen_success_calls = 0
        img_gen_fail_calls = 0
        img_gen_total_duration = 0.0
        img_gen_prompt_token_counts = []
        os.makedirs(save_dir, exist_ok=True)

        with tqdm(total=worker_trajs_count, desc="Worker Rollout Progress", unit="task") as pbar:
            current_iteration = 0
            max_iterations = self.config.max_gen_round
            max_prompt_len_allowed = int(self.config.max_model_len or self.config.prompt_length + self.config.response_length)
            while current_iteration < max_iterations and len(to_generate) > 0:
                idx_to_gen = []
                idx_overlength = []
                for i in to_generate:
                    if len(vllm_inputs[i]['prompt_token_ids']) >= max_prompt_len_allowed:
                        idx_overlength.append(i)
                    else:
                        idx_to_gen.append(vllm_inputs[i])

                if idx_overlength:
                    print(
                        f"[Round #{current_iteration}] Skipping {len(idx_overlength)} trajs "
                        f"exceeding max_model_len ({max_prompt_len_allowed}): "
                        f"lengths={[len(vllm_inputs[i]['prompt_token_ids']) for i in idx_overlength]}"
                    )
                    for x in idx_overlength:
                        to_generate.remove(x)
                    pbar.update(len(idx_overlength))

                if len(idx_to_gen) == 0:
                    current_iteration += 1
                    break

                print(
                    f"[Round #{current_iteration} Rollout START] For THIS round, We hava {len(idx_to_gen)} trajs to complete ..."
                )

                with self.update_sampling_params(n=1):
                    outputs = self.inference_engine.generate(
                        prompts=idx_to_gen, sampling_params=self.sampling_params, use_tqdm=False
                    )

                response = []
                for output in outputs:
                    for sample_id in range(len(output.outputs)):
                        _token_ids = output.outputs[sample_id].token_ids
                        filtered_token_ids = [token_id for token_id in _token_ids if token_id <= 151664]
                        if 151645 not in filtered_token_ids:
                            filtered_token_ids[-1] = 151645

                        response.append(filtered_token_ids)

                assert len(to_generate) == len(response)

                idx_to_remove = []
                id_paint_query_mapping = {}
                to_paint = []
                reached_limit_count = 0
                for i_gen, response_ in zip(to_generate, response):
                    response_ = list(response_)
                    vllm_inputs[i_gen]['prompt_token_ids'] += response_
                    multi_turn_response_mask[i_gen].append(
                        torch.ones(len(response_), dtype=attention_mask.dtype, device=attention_mask.device)
                    )

                    decoded_resp_ = self.tokenizer.decode(response_, skip_special_tokens=True)

                    if re.search(r'<finish>.*?</finish>', decoded_resp_, re.DOTALL):
                        idx_to_remove.append(i_gen)
                        continue

                    paint_match = re.search(r'<paint>(.*?)</paint>', decoded_resp_, re.DOTALL)
                    if paint_match:
                        assert str(i_gen) not in id_paint_query_mapping.keys()
                        if id_img_draw_cnt[i_gen] >= max_img_draw_round or current_iteration == max_iterations - 1:
                            if len(response_) > 0:
                                vllm_inputs[i_gen]['prompt_token_ids'] = vllm_inputs[i_gen]['prompt_token_ids'][:-len(response_)]
                                multi_turn_response_mask[i_gen].pop()

                            def _build_synth_last_ids():
                                reasoning_m = re.search(r'<reasoning>(.*?)</reasoning>', decoded_resp_, re.DOTALL)
                                reasoning_raw = reasoning_m.group(1).strip() if reasoning_m else ""
                                reasoning_txt = re.sub(r'<[^>]+>', '', reasoning_raw).strip() or "Reached maximum paint rounds."

                                text_without_reasoning = re.sub(r'<reasoning>.*?</reasoning>', '', decoded_resp_, flags=re.DOTALL)
                                assess_m = re.search(r'<assess>(.*?)</assess>', text_without_reasoning, re.DOTALL)
                                assess_raw = assess_m.group(1).strip() if assess_m else ""
                                assess_txt = re.sub(r'<[^>]+>', '', assess_raw).strip() or "Submitting current best image."

                                synth_last = (
                                    f"<reasoning>{reasoning_txt}</reasoning>"
                                    f"<assess>{assess_txt}</assess>"
                                    f"<finish>over</finish>"
                                )
                                ids = self.tokenizer(synth_last, add_special_tokens=False)["input_ids"]
                                if 151645 not in ids:
                                    ids.append(151645)
                                return ids

                            is_last_iter = (current_iteration == max_iterations - 1)

                            need_synth_last = is_last_iter or id_force_finish_injected[i_gen]

                            if not need_synth_last:
                                force_finish_msg = (
                                    "<|im_start|>user\n"
                                    "Since this is the final round, you must output <finish>over</finish> to complete the task and must not output <paint>.\n"
                                    "Please output in the following format:\n"
                                    "<reasoning>your reasoning process here</reasoning>\n"
                                    "<assess>your visual assessment here</assess>\n"
                                    "<finish>over</finish>\n"
                                    "<|im_end|>\n<|im_start|>assistant\n"
                                )
                                force_finish_ids = self.tokenizer.encode(force_finish_msg)

                                projected_len = len(vllm_inputs[i_gen]['prompt_token_ids']) + len(force_finish_ids)
                                if projected_len >= max_prompt_len_allowed:
                                    need_synth_last = True
                                    synth_reason = (
                                        f"force-finish injection would exceed max_model_len "
                                        f"(projected={projected_len} >= {max_prompt_len_allowed})"
                                    )
                                else:
                                    vllm_inputs[i_gen]['prompt_token_ids'] += force_finish_ids
                                    multi_turn_response_mask[i_gen].append(
                                        torch.zeros(len(force_finish_ids), dtype=attention_mask.dtype, device=attention_mask.device)
                                    )
                                    id_force_finish_injected[i_gen] = True
                                    reached_limit_count += 1
                                    print(
                                        f"[Round #{current_iteration}] traj {i_gen} hit paint limit "
                                        f"(cnt={id_img_draw_cnt[i_gen]}>={max_img_draw_round}), "
                                        f"injected force-finish prompt, will re-generate next iter"
                                    )
                                    continue

                            synth_ids = _build_synth_last_ids()
                            vllm_inputs[i_gen]['prompt_token_ids'] += synth_ids
                            multi_turn_response_mask[i_gen].append(
                                torch.ones(len(synth_ids), dtype=attention_mask.dtype, device=attention_mask.device)
                            )
                            idx_to_remove.append(i_gen)
                            reached_limit_count += 1
                            if is_last_iter:
                                reason = "reached max_gen_round last iter, no next vLLM chance"
                            elif id_force_finish_injected[i_gen]:
                                reason = "STILL output <paint> after force-finish"
                            else:
                                reason = synth_reason
                            print(
                                f"[Round #{current_iteration}] traj {i_gen} hit paint limit "
                                f"(cnt={id_img_draw_cnt[i_gen]}>={max_img_draw_round}); "
                                f"synthesized last response to end trajectory ({reason})"
                            )
                            continue
                        paint_prompt = None
                        for match in re.finditer(r'<paint>(.*?)</paint>', decoded_resp_, re.DOTALL):
                            paint_prompt = match.group(1).strip()
                        if paint_prompt:
                            id_paint_query_mapping[str(i_gen)] = paint_prompt
                            id_img_draw_cnt[i_gen] += 1
                            to_paint.append(i_gen)
                        else:
                            print(f"[Round #{current_iteration} Rollout ERROR] Empty <paint> tag, traj {i_gen} will be removed.")
                            idx_to_remove.append(i_gen)
                    else:
                        idx_to_remove.append(i_gen)

                print(
                f"[Round #{current_iteration} Rollout Paint Trigger] "
                f"For THIS round, we need to generate images for {len(id_paint_query_mapping)} trajs ..."
            )
                if reached_limit_count > 0:
                    print(f"[Round #{current_iteration}] {reached_limit_count} trajs reached max_img_draw_round {max_img_draw_round}")

                newly_completed = len(idx_to_remove)
                for x in idx_to_remove:
                    to_generate.remove(x)

                print(
                    f"[Round #{current_iteration} Rollout END] Completed {newly_completed} trajs, "
                    f"{len(to_generate)} trajs remaining for next round ..."
                )

                pbar.update(newly_completed)

                if len(to_generate) == 0:
                    current_iteration += 1
                    break

                if len(to_paint) == 0:
                    current_iteration += 1
                    continue

                generate_result = []

                if not self.config.generate.parallel_tool_call:
                    for i_todo in tqdm(to_paint, desc=f"[Round #{current_iteration} Generating Progress]"):
                        assert str(i_todo) in id_paint_query_mapping.keys()
                        paint_prompt = id_paint_query_mapping[str(i_todo)]

                        tool_returned_str, tool_returned_images, tool_stat = call_image_generate(
                            prompt=paint_prompt,
                        )
                        generate_result.append((tool_returned_str, tool_returned_images, tool_stat))
                    success_count = sum(1 for _, _, s in generate_result if s["success"])
                    print(f"[Round #{current_iteration} Generate END] success: {success_count}/{len(to_paint)}")

                else:
                    print(f"[Round #{current_iteration} Generate START] Batch generating {len(to_paint)} images (max_concurrent={os.environ.get('IMAGE_GEN_MAX_CONCURRENT', '16')}) ...")
                    prompts_batch = [id_paint_query_mapping[str(i)] for i in to_paint]
                    generate_result = call_image_generate_batch(prompts=prompts_batch)
                    success_count = sum(1 for _, _, s in generate_result if s["success"])
                    print(f"[Round #{current_iteration} Generate END] success: {success_count}/{len(to_paint)}")

                for _, _, s in generate_result:
                    img_gen_total_calls += 1
                    if s["success"]:
                        img_gen_success_calls += 1
                    else:
                        img_gen_fail_calls += 1
                    img_gen_total_duration += s.get("duration", 0.0)
                    prompt_tokens = len(self.tokenizer.encode(s["prompt"]))
                    img_gen_prompt_token_counts.append(prompt_tokens)

                to_paint_ = to_paint.copy()

                for i_gen_, generate_result_ in zip(to_paint_, generate_result):
                    generate_result_txt, generate_result_img, tool_stat = generate_result_

                    if tool_stat["success"] and generate_result_img:
    
                        prompt_hash = hashlib.md5(tool_stat["prompt"].encode()).hexdigest()[:8]

                        image_save_path = os.path.join(
                            save_dir,
                            f"rank{rank:02d}_round_{current_iteration:02d}_{prompt_hash}.png"
                        )
                        generate_result_img[0].save(image_save_path)  

                        if 'image_urls' not in non_tensor_batch:
                            non_tensor_batch['image_urls'] = np.array(
                                [None] * (batch_size * n), dtype=object
                            )
                        non_tensor_batch['image_urls'][i_gen_] = image_save_path

                        if 'image_urls_per_round' not in non_tensor_batch:
                            non_tensor_batch['image_urls_per_round'] = np.empty(batch_size * n, dtype=object)
                            for _k in range(batch_size * n):
                                non_tensor_batch['image_urls_per_round'][_k] = []
                        _cur_list = non_tensor_batch['image_urls_per_round'][i_gen_]
                        if not isinstance(_cur_list, list):
                            _cur_list = list(_cur_list)
                        _cur_list.append(image_save_path)
                        non_tensor_batch['image_urls_per_round'][i_gen_] = _cur_list

                        print(
                            f"[Round #{current_iteration}] traj {i_gen_} image saved -> {image_save_path}"
                        )

                    if (
                        "[Image Generation Results]" in generate_result_txt
                        and "There is an error" not in generate_result_txt
                        and self.user_prompt_after_image_generate is not None
                    ):
                        all_context = self.tokenizer.decode(
                            vllm_inputs[i_gen_]['prompt_token_ids'],
                            skip_special_tokens=False
                        )
                        
                        next_user_message = (
                            generate_result_txt
                            + self.user_prompt_after_image_generate
                        )
                        next_user_message = (
                        "<|im_start|>user\n"
                        + next_user_message
                        + "<|im_end|>\n<|im_start|>assistant\n"
                    )
                    else:
                        next_user_message = (
                            "<|im_start|>user\n"
                            + generate_result_txt
                            + "<|im_end|>\n<|im_start|>assistant\n"
                        )

                    next_turn_prompt_ids = self.tokenizer.encode(next_user_message)
       
                    if generate_result_img:
                        if 'multi_modal_data' not in vllm_inputs[i_gen_]:
                            vllm_inputs[i_gen_]['multi_modal_data'] = {'image': []}

                        image_inputs = self.processor.image_processor(
                            generate_result_img, return_tensors='pt'
                        )
                        image_grid_thw = image_inputs['image_grid_thw']
                        if image_grid_thw is not None:
                            merge_length = self.processor.image_processor.merge_size ** 2
                            replicate_count = image_grid_thw[0].prod() // merge_length
                            image_pad_token = 151655

                        vllm_inputs[i_gen_]['prompt_token_ids'] += next_turn_prompt_ids
                        vllm_inputs[i_gen_]['multi_modal_data']['image'].append(generate_result_img[0])
                        generate_tool_return_images[i_gen_].append(generate_result_img[0])

                    else:
                        vllm_inputs[i_gen_]['prompt_token_ids'] += next_turn_prompt_ids

                    multi_turn_response_mask[i_gen_].append(
                        torch.zeros(
                            len(next_turn_prompt_ids),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        )
                    )

                current_iteration += 1

        response = []
        response_generation_mask = []

        for i_ in range(batch_size * n):
            all_response_masks = torch.cat(multi_turn_response_mask[i_][1:], dim=0)
            resp_mask_device = all_response_masks.device

            first_round_prompt_length = prefix_prompt_lengths[i_]
            response_after_prompt = vllm_inputs[i_]['prompt_token_ids'][first_round_prompt_length:]

            if generate_tool_return_images[i_]:
                generated_image_inputs = self.processor.image_processor(
                    generate_tool_return_images[i_], return_tensors='pt'
                )
                generated_image_grid_thw = generated_image_inputs['image_grid_thw']

                if generated_image_grid_thw is not None:
                    merge_length = self.processor.image_processor.merge_size ** 2
                    image_pad_token, magic_num = 151655, 654321
                    all_response_masks = all_response_masks.tolist()

                    index = 0 
                    while image_pad_token in response_after_prompt:
                        pos = response_after_prompt.index(image_pad_token)
                        replicate_count = generated_image_grid_thw[index].prod() // merge_length
                        response_after_prompt[pos: pos + 1] = [magic_num] * replicate_count
                        all_response_masks[pos: pos + 1] = [0] * replicate_count
                        index += 1

                    response_after_prompt = [
                        image_pad_token if x == magic_num else x
                        for x in response_after_prompt
                    ]
                    all_response_masks = torch.tensor(
                        all_response_masks, dtype=torch.int64, device=resp_mask_device
                    )

            response_generation_mask.append(all_response_masks)
            all_response = torch.tensor(response_after_prompt, device=idx.device, dtype=idx.dtype)
            response.append(all_response)
            assert (
                response[i_].shape[0] == response_generation_mask[i_].shape[0]
            ), (
                f"shape mismatched | "
                f"response[i_]: {response[i_].shape[0]} | "
                f"response_generation_mask[i_]: {response_generation_mask[i_].shape[0]}"
            )

        assert len(response) == len(response_generation_mask), \
            "length mismatched between response and response_generation_mask!"

        response = pad_to_max_stack(
            response, self.pad_token_id, dim=0
        )
        response_generation_mask = pad_to_max_stack(
            response_generation_mask, 0, dim=0
        )
        assert all([
            response.size(dim) == response_generation_mask.size(dim)
            for dim in range(response.ndim)
        ])

        if response.shape[1] > self.config.response_length_total:
            response = response[:, :self.config.response_length_total]
            response_generation_mask = response_generation_mask[:, :self.config.response_length_total]
        elif response.shape[1] < self.config.response_length_total:
            response = pad_sequence_to_length(
                response, self.config.response_length_total, self.pad_token_id
            )
            response_generation_mask = pad_sequence_to_length(
                response_generation_mask, self.config.response_length_total, 0
            )

        if n > 1 and do_sample:
            idx = _repeat_interleave(idx, n)
            attention_mask = _repeat_interleave(attention_mask, n)
            position_ids = _repeat_interleave(position_ids, n)
            batch_size = batch_size * n

            input_prompt_generation_mask = _repeat_interleave(
                input_prompt_generation_mask, n
            )

        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_attention_mask = get_final_eos_mask(
            response_id=response, eos_token=[151645], dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        multi_turn_response_mask = torch.cat(
            [input_prompt_generation_mask, response_generation_mask], dim=-1
        )

        batch = TensorDict(
            {
                'prompts': idx.contiguous(),
                'responses': response.contiguous(),
                'input_ids': seq.contiguous(),
                'attention_mask': attention_mask.contiguous(),
                'position_ids': position_ids.contiguous(),
                'multi_turn_response_mask': multi_turn_response_mask.contiguous(),
            },
            batch_size=batch_size,
        )

        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        _img_gen_stats = {
            'total_calls': img_gen_total_calls,
            'success_calls': img_gen_success_calls,
            'fail_calls': img_gen_fail_calls,
            'total_duration': img_gen_total_duration,
            'prompt_token_counts': img_gen_prompt_token_counts,
        }
        non_tensor_batch['_img_gen_stats'] = np.array(
            [_img_gen_stats] * batch_size, dtype=object
        )

        print(">>> vllm_rollout_spmd Rollout Ends ...")
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)