import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import auc


class ROCMetric(object):
    def __init__(self, bins=100):
        self.bins = bins
        self.reset()

    def update(self, pred, label):
        max_pred = np.max(pred)
        if max_pred <= 0:
            raise ValueError("Cannot normalize prediction: all values are non-positive")
        pred = pred / max_pred # normalize output to 0-1
        label = label.astype(np.uint8)

        # analysis target number
        num_labels, labels, _, centroids = cv2.connectedComponentsWithStats(label)
        #assert num_labels > 1
        if(num_labels <=1):
            return

        # get masks and update background area and targets number
        back_mask = labels == 0
        tmp_back_area = np.sum(back_mask)
        self.background_area += tmp_back_area
        self.target_nums += (num_labels - 1)

        for ibin in range(self.bins + 1):
            thre = ibin / self.bins
            pred_binary = pred >= thre

            # update false detection
            tmp_false_detect = np.sum(np.logical_and(back_mask, pred_binary))
            if tmp_false_detect > tmp_back_area:
                raise ValueError(
                    f"False detections ({tmp_false_detect}) exceed background area ({tmp_back_area}); "
                    "prediction and label shapes are probably mismatched."
                )
            self.false_detect[ibin] += tmp_false_detect

            # update true detection, there maybe multiple targets
            for t in range(1, num_labels):
                target_mask = labels == t
                self.true_detect[ibin] += np.sum(np.logical_and(target_mask, pred_binary)) > 0

    def get(self):
        if self.background_area == 0 or self.target_nums == 0:
            raise RuntimeError(
                "No labelled targets were accumulated; call update() with non-empty labels before get()"
            )
        fpr = self.false_detect / self.background_area  # X axis
        tpr = self.true_detect / self.target_nums       # Y axis
        return fpr, tpr, auc(fpr, tpr)

    def get_all(self):
        return self.false_detect, self.background_area, self.true_detect, self.target_nums

    def reset(self):
        self.false_detect = np.zeros(self.bins+1)
        self.true_detect = np.zeros(self.bins+1)
        self.background_area = 0
        self.target_nums = 0