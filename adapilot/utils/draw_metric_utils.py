
import numpy as np

def aggregate_draw_metrics(batch) -> dict:
    result = {}

    if '_img_gen_stats' in batch.non_tensor_batch:
        stats = batch.non_tensor_batch['_img_gen_stats'][0]
        if isinstance(stats, dict):
            total = stats.get('total_calls', 0)
            result['draw/img_gen_api/total_calls'] = total
            result['draw/img_gen_api/success_calls'] = stats.get('success_calls', 0)
            result['draw/img_gen_api/fail_calls'] = stats.get('fail_calls', 0)
            result['draw/img_gen_api/success_rate'] = (
                stats.get('success_calls', 0) / max(total, 1)
            )
            result['draw/img_gen_api/total_duration'] = stats.get('total_duration', 0.0)
            result['draw/img_gen_api/avg_duration'] = (
                stats.get('total_duration', 0.0) / max(total, 1)
            )
            token_counts = stats.get('prompt_token_counts', [])
            if token_counts:
                result['draw/paint_prompt_tokens/mean'] = float(np.mean(token_counts))
                result['draw/paint_prompt_tokens/max'] = float(np.max(token_counts))
                result['draw/paint_prompt_tokens/min'] = float(np.min(token_counts))
                result['draw/paint_prompt_tokens/total'] = int(np.sum(token_counts))

    if 'extra_info' not in batch.non_tensor_batch:
        return result

    all_metrics = []
    for ei in batch.non_tensor_batch['extra_info']:
        if isinstance(ei, dict) and '_draw_metrics' in ei:
            all_metrics.append(ei['_draw_metrics'])

    if not all_metrics:
        return result

    n = len(all_metrics)

    scalar_keys = [
        'format_score', 'paint_count', 'monotonic_ratio',
        'final_round_score', 'content_reward',
        'budget_score', 'budget_bonus', 'reward',
    ]
    for key in scalar_keys:
        vals = [m.get(key, 0.0) for m in all_metrics]
        if vals:
            result[f'draw/{key}/mean'] = float(np.mean(vals))
            result[f'draw/{key}/max'] = float(np.max(vals))
            result[f'draw/{key}/min'] = float(np.min(vals))

    metric_names = ['clip_t', 'hps', 'aesthetic', 'gdino', 'ocr', 'realism', 'physics']
    for name in metric_names:
        key = f'final_{name}'
        vals = [m[key] for m in all_metrics if key in m]
        if vals:
            result[f'draw/{key}/mean'] = float(np.mean(vals))
            result[f'draw/{key}/max'] = float(np.max(vals))
            result[f'draw/{key}/min'] = float(np.min(vals))

    format_perfect = sum(1 for m in all_metrics if m.get('format_score', 0) >= 1.0)
    result['draw/format_perfect_ratio'] = format_perfect / n

    paint_counts = [m.get('paint_count', 0) for m in all_metrics]
    result['draw/paint_count/mean'] = float(np.mean(paint_counts))

    mono_vals = [m.get('monotonic_ratio', 0.0) for m in all_metrics]
    result['draw/monotonic_positive_ratio'] = sum(1 for v in mono_vals if v > 0) / n

    max_rounds = max(len(m.get('round_scores', [])) for m in all_metrics)
    for r_idx in range(min(max_rounds, 10)):
        vals = [
            m['round_scores'][r_idx]
            for m in all_metrics
            if len(m.get('round_scores', [])) > r_idx
        ]
        if vals:
            result[f'draw/round_{r_idx}_score/mean'] = float(np.mean(vals))
            result[f'draw/round_{r_idx}_score/count'] = len(vals)

    result['draw/num_samples'] = n

    return result