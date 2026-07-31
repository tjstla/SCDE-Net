import threading

import numpy
import numpy as np
import torch
import torch.nn.functional as F

__all__ = ['SegmentationMetricTPFNFP']

def get_miou_prec_recall_fscore(total_tp, total_fp, total_fn):
    miou = 1.0 * total_tp / (np.spacing(1) + total_tp + total_fp + total_fn)
    prec = 1.0 * total_tp / (np.spacing(1) + total_tp + total_fp)
    recall = 1.0 * total_tp / (np.spacing(1) + total_tp + total_fn)
    fscore = 2.0 * prec * recall / (np.spacing(1) + prec + recall)

    return miou, prec, recall, fscore

class SegmentationMetricTPFNFP(object):
    """Computes pixAcc and mIoU metric scroes
    """

    def __init__(self, nclass):
        self.nclass = nclass
        self.lock = threading.Lock()
        self.reset()

    def update(self, labels, preds):
        def evaluate_worker(self, label, pred):
            tp, fp, fn = batch_tp_fp_fn(pred, label, self.nclass)
            with self.lock:
                self.total_tp += tp
                self.total_fp += fp
                self.total_fn += fn
                self.num_updates += 1
            return

        if isinstance(preds, torch.Tensor):
            preds = (preds.detach().numpy() > 0).astype('int64')  # P
            labels = labels.numpy().astype('int64')  # T
            evaluate_worker(self, labels, preds)
        elif isinstance(preds, (list, tuple)):
            threads = [threading.Thread(target=evaluate_worker,
                                        args=(self, label, pred),
                                        )
                       for (label, pred) in zip(labels, preds)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        #elif preds.dtype == numpy.uint8:
        elif isinstance(preds, np.ndarray):
            pred_max = np.max(preds)
            label_max = np.max(labels)
            if pred_max <= 0:
                raise ValueError("Cannot normalize predictions: all values are non-positive")
            if label_max <= 0:
                raise ValueError("Cannot normalize labels: the label contains no foreground pixels")
            preds = ((preds / pred_max) > 0.5).astype('int64')  # P
            labels = (labels / label_max).astype('int64')  # T
            evaluate_worker(self, labels, preds)
        else:
            raise TypeError(
                f"Unsupported prediction type: {type(preds)}. "
                "Expected torch.Tensor, numpy.ndarray, list or tuple."
            )

    def get_all(self):
        return self.total_tp, self.total_fp, self.total_fn

    def get(self):
        if self.num_updates == 0:
            raise RuntimeError("No samples were accumulated; call update() before get()")
        return get_miou_prec_recall_fscore(self.total_tp, self.total_fp, self.total_fn)

    def reset(self):
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.num_updates = 0
        return

def batch_tp_fp_fn(predict, target, nclass):
    """Batch Intersection of Union
    Args:
        predict: input 4D tensor
        target: label 3D tensor
        nclass: number of categories (int)
    """

    mini = 1
    maxi = nclass
    nbins = nclass

    # predict = (output.detach().numpy() > 0).astype('int64')  # P
    # target = target.numpy().astype('int64')  # T
    intersection = predict * (predict == target)  # TP

    # areas of intersection and union
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))

    # areas of TN FP FN
    area_tp = area_inter[0]
    area_fp = area_pred[0] - area_inter[0]
    area_fn = area_lab[0] - area_inter[0]

    # area_union = area_pred + area_lab - area_inter
    if area_fn < 0 or area_fp < 0:
        raise ValueError(
            f"Inconsistent confusion counts (tp={area_tp}, fp={area_fp}, fn={area_fn}); "
            "predictions and labels are probably not binary maps of the same shape."
        )
    return area_tp, area_fp, area_fn

