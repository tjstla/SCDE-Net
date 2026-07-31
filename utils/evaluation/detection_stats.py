import cv2
import numpy as np

__all__ = ['normalize_pred', 'label_targets', 'count_false_detect', 'count_true_detect',
           'DetectionCounter']


def normalize_pred(pred):
    """Normalize a prediction map to 0-1."""
    return pred / np.max(pred)


def label_targets(label):
    """Label connected target components.

    Returns (num_labels, labels) or None when the label map holds no target.
    """
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(label.astype(np.uint8))
    if num_labels <= 1:
        return None
    return num_labels, labels


def count_false_detect(back_mask, pred_binary):
    return np.sum(np.logical_and(back_mask, pred_binary))


def count_true_detect(labels, num_labels, pred_binary):
    """Number of targets with at least one predicted pixel."""
    return sum(np.sum(np.logical_and(labels == t, pred_binary)) > 0
               for t in range(1, num_labels))


class DetectionCounter(object):
    """Bookkeeping of background area / target number shared by detection metrics."""

    def __init__(self):
        self.reset()

    def _zero_detections(self):
        return 0, 0

    def reset(self):
        self.false_detect, self.true_detect = self._zero_detections()
        self.background_area = 0
        self.target_nums = 0

    def get_all(self):
        return self.false_detect, self.background_area, self.true_detect, self.target_nums

    def accumulate_background(self, back_mask, num_labels):
        """Update background area / target count, returning the background area of this sample."""
        back_area = np.sum(back_mask)
        self.background_area += back_area
        self.target_nums += num_labels - 1
        return back_area
