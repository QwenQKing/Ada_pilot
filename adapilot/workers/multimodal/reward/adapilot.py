import json

import torch
from verl import DataProto


class AdaPilotRewardManager:

    def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine

    def extract_responses_list(
        self,
        tokenizer,
        input_ids: torch.Tensor,
        multi_turn_response_mask: torch.Tensor,
    ) -> list:
        diff = torch.diff(multi_turn_response_mask, prepend=torch.tensor([0], device=multi_turn_response_mask.device))
        starts = torch.where(diff == 1)[0]
        mask_appended = torch.cat(
            [multi_turn_response_mask, torch.tensor([0], device=multi_turn_response_mask.device)], dim=0
        )
        diff_end = torch.diff(mask_appended)
        ends = torch.where(diff_end == -1)[0] - 1
        segments = []
        for s, e in zip(starts, ends):
            segments.append(input_ids[s : e + 1].tolist())

        decoded_responses = tokenizer.batch_decode(segments, skip_special_tokens=True)
        return decoded_responses

    def _extract_item_fields(self, data_item):
        prompt_ids    = data_item.batch['prompts']
        prompt_length = prompt_ids.shape[-1]
        valid_prompt_length   = data_item.batch['attention_mask'][:prompt_length].sum()
        valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()

        valid_prompt_ids   = prompt_ids[-valid_prompt_length:]
        response_ids       = data_item.batch['responses']

        prompt_str   = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        response_str = self.tokenizer.decode(response_ids[:valid_response_length], skip_special_tokens=True)
        response_str = [response_str]

        if 'multi_turn_response_mask' in data_item.batch:
            response_str = self.extract_responses_list(
                self.tokenizer,
                data_item.batch['input_ids'],
                data_item.batch['multi_turn_response_mask'],
            )

        ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
        if isinstance(ground_truth, str):
            ground_truth = [ground_truth]
        if 'candidate_answers' in data_item.non_tensor_batch['reward_model']:
            ca = data_item.non_tensor_batch['reward_model']['candidate_answers']
            if isinstance(ca, list):
                ground_truth += ca
            elif isinstance(ca, str):
                ground_truth += json.loads(ca)
            else:
                raise TypeError(f"candidate_answers must be list or str, got {type(ca)}")
        ground_truth = [g for g in ground_truth if isinstance(g, str)]

        return dict(
            prompt_str            = prompt_str,
            response_str          = response_str,
            ground_truth          = ground_truth,
            valid_response_length = valid_response_length.item(),
            data_source           = data_item.non_tensor_batch['data_source'],
            extra_info            = data_item.non_tensor_batch.get('extra_info', None),
            image_url             = data_item.non_tensor_batch.get('image_urls', None),
            image_urls_per_round  = data_item.non_tensor_batch.get('image_urls_per_round', None),
            caption               = data_item.non_tensor_batch['reward_model'].get('caption', None),
        )

    def __call__(self, data: DataProto):
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        already_print_data_sources = {}

        all_fields = [self._extract_item_fields(data[i]) for i in range(len(data))]

        from adapilot.utils.reward_score_mm.draw_score import DATASET_PRIMARY_METRICS, compute_scores_batch

        scores = [0.0] * len(all_fields)

        batch_extras = []
        for f in all_fields:
            e = dict(f['extra_info'] or {})
            e['data_source'] = f['data_source']
            batch_extras.append(e)

        batch_scores = compute_scores_batch(
            predictions               = [f['response_str']          for f in all_fields],
            ground_truths             = [f['ground_truth']           for f in all_fields],
            image_urls                = [f['image_url']              for f in all_fields],
            captions                  = [f['caption']                for f in all_fields],
            extra_infos               = batch_extras,
            image_urls_per_round_list = [f.get('image_urls_per_round') for f in all_fields],
        )
        for j in range(len(all_fields)):
            scores[j] = batch_scores[j]
            if '_draw_metrics' in batch_extras[j]:
                data.non_tensor_batch['extra_info'][j]['_draw_metrics'] = batch_extras[j]['_draw_metrics']

        for i, f in enumerate(all_fields):
            vrl = f['valid_response_length']
            reward_tensor[i, vrl - 1] = scores[i]

            ds = f['data_source']
            if ds not in already_print_data_sources:
                already_print_data_sources[ds] = 0
            if already_print_data_sources[ds] < self.num_examine:
                already_print_data_sources[ds] += 1
                print("[prompt]",       f['prompt_str'])
                print("[response]",     f['response_str'])
                print("[ground_truth]", f['ground_truth'])
                print("[score]",        scores[i])

        return reward_tensor
