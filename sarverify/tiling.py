import cv2
import numpy as np


def tile_starts(n, tile, step):
    if n <= tile:
        return [0]
    out = list(range(0, n - tile, step))
    out.append(n - tile)
    return out


def detect_tiled(det, bgr, tile=960, step=800, conf=None, iou=0.45):
    h, w = bgr.shape[:2]
    boxes, scores = [], []
    for y in tile_starts(h, tile, step):
        for x in tile_starts(w, tile, step):
            crop = bgr[y:y + tile, x:x + tile]
            for x1, y1, x2, y2, sc in det.detect(crop, conf, upscale=False):
                boxes.append([x1 + x, y1 + y, x2 - x1, y2 - y1])
                scores.append(sc)
    if not boxes:
        return []
    idx = cv2.dnn.NMSBoxes(boxes, scores, conf if conf is not None else det.conf, iou)
    out = []
    for i in np.array(idx).flatten():
        bx, by, bw, bh = boxes[i]
        out.append((bx, by, bx + bw, by + bh, scores[i]))
    return out
