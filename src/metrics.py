# With direct inspiration from https://github.com/neptune-ml/open-solution-ship-detection/blob/master/src/metrics.py

from functools import partial

import numpy as np
from pycocotools import mask as cocomask

from argus.metrics import Metric


EPS = 1e-9


def rle_from_binary(prediction):
    prediction = np.asfortranarray(prediction)
    return cocomask.encode(prediction)


def binary_from_rle(rle):
    return cocomask.decode(rle)

def get_segmentations(labeled):
    nr_true = labeled.max()
    segmentations = []
    for i in range(1, nr_true + 1):
        msk = labeled == i
        segmentation = rle_from_binary(msk.astype('uint8'))
        segmentation['counts'] = segmentation['counts'].decode("UTF-8")
        segmentations.append(segmentation)
    return segmentations


def iou(gt, pred):
    gt[gt > 0] = 1.
    pred[pred > 0] = 1.
    intersection = gt * pred
    union = gt + pred
    union[union > 0] = 1.
    intersection = np.sum(intersection)
    union = np.sum(union)
    if union == 0:
        union = EPS
    return intersection / union


def compute_ious(gt, predictions):
    gt_ = get_segmentations(gt)
    predictions_ = get_segmentations(predictions)

    if len(gt_) == 0 and len(predictions_) == 0:
        return np.ones((1, 1))
    elif len(gt_) != 0 and len(predictions_) == 0:
        return np.zeros((1, 1))
    else:
        iscrowd = [0 for _ in predictions_]
        ious = cocomask.iou(gt_, predictions_, iscrowd)
        if not np.array(ious).size:
            ious = np.zeros((1, 1))
        return ious


def compute_precision_at(ious, threshold):
    mx1 = np.max(ious, axis=1)
    mx2 = np.max(ious, axis=0)
    tp = np.sum(mx2 >= threshold)
    fp = np.sum(mx2 < threshold)
    fn = np.sum(mx1 < threshold)
    return float(tp) / (tp + fp + fn)

def compute_f_beta_at(ious, threshold, beta):
    mx1 = np.max(ious, axis=1)
    mx2 = np.max(ious, axis=0)
    tp = np.sum(mx2 >= threshold)
    fp = np.sum(mx2 < threshold)
    fn = np.sum(mx1 < threshold)
    return (1 + beta**2) * tp / ((1 + beta**2) * tp + (beta**2) * fn + fp)


def compute_eval_metric(gt, predictions, metric_to_average='precision'):
    thresholds = [0.1, 0.2, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    ious = compute_ious(gt, predictions)
    if metric_to_average == 'precision':
        metric_function = compute_precision_at
    elif metric_to_average[0] == 'f':
        beta = float(metric_to_average[1:])
        metric_function = partial(compute_f_beta_at, beta=beta)
    else:
        raise NotImplementedError
    metric_per_image = [metric_function(ious, th) for th in thresholds]
    return sum(metric_per_image) / len(metric_per_image)


class ShipIOUT(Metric):
    name = 'iout'
    better = 'max'

    def reset(self):
        self.iout_sum = 0
        self.count = 0

    def update(self, output: dict):
		# Move to cpu only the basic mask
        preds = output['prediction'][:, 0, :, :].cpu().numpy().astype(np.uint8)
        trgs = output['target'][:, 0, :, :].cpu().numpy().astype(np.uint8)

        for i in range(trgs.shape[0]):
            pred = preds[i, :, :]
            trg = trgs[i, :, :]
            self.iout_sum += compute_eval_metric(trg, pred, metric_to_average='f2')
            self.count += 1

    def compute(self):
        return self.iout_sum / self.count
