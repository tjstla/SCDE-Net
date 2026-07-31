from utils.evaluation.detection_stats import (DetectionCounter, count_false_detect,
                                              count_true_detect, label_targets, normalize_pred)


class my_PD_FA(DetectionCounter):
    def update(self, pred, label):
        pred = normalize_pred(pred)

        targets = label_targets(label)
        if targets is None:
            return
        num_labels, labels = targets

        back_mask = labels == 0
        back_area = self.accumulate_background(back_mask, num_labels)

        pred_binary = pred > 0.5

        false_detect = count_false_detect(back_mask, pred_binary)
        assert false_detect <= back_area
        self.false_detect += false_detect

        self.true_detect += count_true_detect(labels, num_labels, pred_binary)

    def get(self):
        FA = self.false_detect / self.background_area
        PD = self.true_detect / self.target_nums
        return PD, FA
