from transformers import EvalPrediction
import numpy as np

def compute_metrics(p: EvalPrediction):
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    labels = p.label_ids
    logits = np.array(logits)
    labels = np.array(labels)
    preds = np.argmax(logits, axis=-1)
    valid_indices = (labels != -100)
    valid_preds = preds[valid_indices]
    valid_labels = labels[valid_indices]
    accuracy = np.mean(valid_preds == valid_labels)
    return {'mlm_accuracy': accuracy}

def log_metrics(args, metrics, header=None): # need a log_path in args, needs to be txt file
    def log_nested_dict(d, parent_key=''):
        filtered_results = {}
        for k, v in d.items():
            new_key = f'{parent_key}_{k}' if parent_key else k
            if isinstance(v, dict):
                filtered_results.update(log_nested_dict(v, new_key))
            elif 'runtime' not in k or 'second' not in k:
                filtered_results[new_key] = round(v, 5) if isinstance(v, (float, int)) else v
        return filtered_results

    filtered_results = log_nested_dict(metrics)

    with open(args.log_path, 'a') as f:
        if header is not None:
            f.write(header + '\n')
        for k, v in filtered_results.items():
            f.write(f'{k}: {v}\n')
        f.write('\n')