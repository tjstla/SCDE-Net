import numpy as np
from sklearn.metrics import auc

from utils.evaluation.detection_stats import (DetectionCounter, count_false_detect,
                                              count_true_detect, label_targets, normalize_pred)


class ROCMetric(DetectionCounter):
    def __init__(self, bins=100):
        self.bins = bins
        super().__init__()

    def _zero_detections(self):
        return np.zeros(self.bins + 1), np.zeros(self.bins + 1)

    def update(self, pred, label):
        pred = normalize_pred(pred)

        targets = label_targets(label)
        if targets is None:
            return
        num_labels, labels = targets

        back_mask = labels == 0
        back_area = self.accumulate_background(back_mask, num_labels)

        for ibin in range(self.bins + 1):
            pred_binary = pred >= ibin / self.bins

            false_detect = count_false_detect(back_mask, pred_binary)
            assert false_detect <= back_area
            self.false_detect[ibin] += false_detect

            self.true_detect[ibin] += count_true_detect(labels, num_labels, pred_binary)

    def get(self):
        fpr = self.false_detect / self.background_area  # X axis
        tpr = self.true_detect / self.target_nums       # Y axis
        return fpr, tpr, auc(fpr, tpr)
