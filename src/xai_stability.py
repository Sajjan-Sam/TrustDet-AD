"""Measure EigenCAM saliency stability under image perturbations.

Uses a forward hook on a late convolutional feature map for each detector and
compares saliency crops for matched detections across transforms. EigenCAM is
gradient-free and is applied consistently across detector architectures.
Missing saliency is logged by the scoring pipeline and assigned stability zero."""
import numpy as np
import torch
import cv2


class EigenCAMHook:
    def __init__(self, target_layer):
        self.activations = None
        self.handle = target_layer.register_forward_hook(self._hook)

    def _hook(self, module, inp, output):
        self.activations = output.detach()

    def remove(self):
        self.handle.remove()

    def saliency_map(self):
        """Returns HxW numpy array in [0,1], or None if no activation was captured."""
        if self.activations is None:
            return None
        act = self.activations[0]  # [C,H,W]
        c, h, w = act.shape
        if h * w < 2 or c < 2:
            return None
        A = act.reshape(c, h * w).float().cpu().numpy()
        A = A - A.mean(axis=0, keepdims=True)
        try:
            _, _, Vt = np.linalg.svd(A, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        sal = Vt[0].reshape(h, w)
        mean_act = A.mean(axis=0).reshape(h, w)
        if np.corrcoef(sal.flatten(), mean_act.flatten())[0, 1] < 0:
            sal = -sal
        sal = sal - sal.min()
        if sal.max() > 0:
            sal = sal / sal.max()
        return sal


def get_target_layer(model, detector_name):
    """Returns the nn.Module to hook for EigenCAM, per detector architecture."""
    if detector_name in ("yolov8n", "yolov10n", "rtdetr"):
        return model.model.model[-2]  # last backbone/neck feature block before the detection head
    if detector_name == "fasterrcnn":
        return model.backbone.body.layer4  # last ResNet-50 block before the FPN/RPN/ROI heads
    raise ValueError(f"unknown detector: {detector_name}")


def crop_and_resize(saliency, xyxy_box, img_shape, out_size=32):
    """Crop the saliency map (already resized to img_shape) to a box region and resize to out_size."""
    h_img, w_img = img_shape
    sal_full = cv2.resize(saliency, (w_img, h_img))
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy_box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = sal_full[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (out_size, out_size))


def saliency_similarity(crop_a, crop_b):
    """Cosine similarity between two equal-size saliency crops, clipped to [0,1]."""
    if crop_a is None or crop_b is None:
        return None
    a, b = crop_a.flatten(), crop_b.flatten()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return None
    sim = float(np.dot(a, b) / denom)
    return max(0.0, min(1.0, sim))
