import numpy as np
import cv2

class my_PD_FA(object):
    def __init__(self, ):
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
        if(num_labels <= 1):
            return

        # get masks and update background area and targets number
        back_mask = labels == 0
        tmp_back_area = np.sum(back_mask)
        self.background_area += tmp_back_area
        self.target_nums += (num_labels - 1)


        pred_binary = pred > 0.5

        # update false detection
        tmp_false_detect = np.sum(np.logical_and(back_mask, pred_binary))
        if tmp_false_detect > tmp_back_area:
            raise ValueError(
                f"False detections ({tmp_false_detect}) exceed background area ({tmp_back_area}); "
                "prediction and label shapes are probably mismatched."
            )
        self.false_detect += tmp_false_detect

        # update true detection, there maybe multiple targets
        for t in range(1, num_labels):
            target_mask = labels == t
            self.true_detect += np.sum(np.logical_and(target_mask, pred_binary)) > 0

    def get(self):
        if self.background_area == 0 or self.target_nums == 0:
            raise RuntimeError(
                "No labelled targets were accumulated; call update() with non-empty labels before get()"
            )
        FA = self.false_detect / self.background_area  #
        PD = self.true_detect / self.target_nums       #
        return PD,FA

    def get_all(self):
        return self.false_detect, self.background_area, self.true_detect, self.target_nums

    def reset(self):
        self.false_detect = 0
        self.true_detect = 0
        self.background_area = 0
        self.target_nums = 0