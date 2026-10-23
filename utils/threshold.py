import numpy as np


def get_gate_finalize_threshold(valid_pred, valid_score, valid_label, percentile=65):
    incorrect = valid_score[valid_pred != valid_label]
    if len(incorrect) == 0:
        return 0.0
    return float(np.quantile(incorrect, percentile / 100))


def get_gate_ood_threshold(valid_score, percentile=5):
    return float(np.quantile(valid_score, percentile / 100))
