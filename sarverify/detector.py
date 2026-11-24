import os

import math

import cv2
import numpy as np
import onnxruntime as ort

from .cinema import project_pose

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "sar_person_yolov8n_v2_best_int8.onnx")
INPUT_SIZE = 960


class PersonDetector:
    def __init__(self, path=MODEL_PATH, conf=0.25, iou=0.45):
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.name = self.sess.get_inputs()[0].name
        self.conf = conf
        self.iou = iou

    def detect_raw(self, bgr, conf=None):
        return self.detect(bgr, conf, upscale=False)

    def _prep(self, img, upscale=True):
        h, w = img.shape[:2]
        scale = min(INPUT_SIZE / h, INPUT_SIZE / w)
        if not upscale:
            scale = min(scale, 1.0)
        nh, nw = round(h * scale), round(w * scale)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
        top, left = (INPUT_SIZE - nh) // 2, (INPUT_SIZE - nw) // 2
        canvas[top:top + nh, left:left + nw] = resized
        blob = canvas[:, :, ::-1].astype(np.float32) / 255.0
        return np.ascontiguousarray(blob.transpose(2, 0, 1)[None]), scale, left, top

    def detect(self, bgr, conf=None, upscale=True):
        conf = self.conf if conf is None else conf
        blob, scale, px, py = self._prep(bgr, upscale)
        out = self.sess.run(None, {self.name: blob})[0][0].T
        scores = out[:, 4]
        keep = scores > conf
        out, scores = out[keep], scores[keep]
        if len(out) == 0:
            return []
        cx, cy, w, h = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
        H, W = bgr.shape[:2]
        x1 = np.clip((cx - w / 2 - px) / scale, 0, W)
        y1 = np.clip((cy - h / 2 - py) / scale, 0, H)
        x2 = np.clip((cx + w / 2 - px) / scale, 0, W)
        y2 = np.clip((cy + h / 2 - py) / scale, 0, H)
        boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)
        idx = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf, self.iou)
        if len(idx) == 0:
            return []
        return [(float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]), float(scores[i])) for i in np.array(idx).flatten()]


def detect_scaled(det, bgr, alt, fov, target_ppm, conf=None):
    h, w = bgr.shape[:2]
    if not target_ppm:
        return det.detect(bgr, conf)
    native = (w / 2.0) / np.tan(fov / 2.0) / alt
    k = target_ppm / native
    k = min(k, INPUT_SIZE / max(h, w))
    nw, nh = max(8, int(round(w * k))), max(8, int(round(h * k)))
    small = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
    dets = det.detect_raw(small, conf)
    return [(x1 / k, y1 / k, x2 / k, y2 / k, sc) for x1, y1, x2, y2, sc in dets]


def webots_image(camera):
    w, h = camera.getWidth(), camera.getHeight()
    arr = np.frombuffer(camera.getImage(), dtype=np.uint8).reshape((h, w, 4))
    return np.ascontiguousarray(arr[:, :, :3])


CONF = 0.25
MATCH_RADIUS_M = 3.0
TARGET_HEIGHT = 0.6


def look_flag(det, bgr, cam, yaw, pitch_down, cand_xy, fov, conf=CONF, radius=MATCH_RADIUS_M):
    h, w = bgr.shape[:2]
    dets = det.detect(bgr, conf)
    r = project_pose((cand_xy[0], cand_xy[1], TARGET_HEIGHT), cam, yaw, pitch_down, w, h, fov)
    best = 0.0
    px = None
    if r is not None:
        f = (w / 2.0) / math.tan(fov / 2.0)
        rad = radius * f / r[2]
        px = (r[0], r[1])
        for x1, y1, x2, y2, sc in dets:
            if math.hypot((x1 + x2) / 2 - r[0], (y1 + y2) / 2 - r[1]) <= rad:
                best = max(best, sc)
    return {"flag": best >= conf, "score": best, "dets": dets, "target_px": px}
