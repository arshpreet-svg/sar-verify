import json
import os
import subprocess
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sarverify.detector import PersonDetector
from sarverify.overlay import font
from sarverify.tiling import detect_tiled
from PIL import Image, ImageDraw

SRC = os.environ.get("REAL_CLIPS", os.path.join(ROOT, "runs", "real"))
CLIPS = [("c64.mp4", "DJI_0064.MP4", 17), ("c73.mp4", "DJI_0073.MP4", 17)]
OUT = os.path.join(ROOT, "runs", "real", "real_segment.mp4")
STATS = os.path.join(ROOT, "results", "real_footage.json")
W, H = 1280, 720


def label_frame(frame, dets, name, t, n_total):
    import numpy as np

    vw_, vh_ = 1024, 576
    small = cv2.resize(frame, (vw_, vh_))
    sx, sy = vw_ / frame.shape[1], vh_ / frame.shape[0]
    for x1, y1, x2, y2, sc in dets:
        cx, cy = (x1 + x2) / 2 * sx, (y1 + y2) / 2 * sy
        r = max(16, (x2 - x1) * sx * 1.3)
        cv2.circle(small, (int(cx), int(cy)), int(r), (0, 255, 60), 2)
        cv2.putText(small, f"{sc:.2f}", (int(cx + r + 3), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 60), 2)
    canvas = Image.new("RGB", (W, H), (14, 18, 28))
    canvas.paste(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)), (0, 72))
    d = ImageDraw.Draw(canvas)
    d.text((24, 16), "REAL DRONE FOOTAGE + OUR PERSON DETECTOR", font=font(30, True), fill=(255, 255, 255))
    d.text((1034, 82), "ZOOM ON TOP", font=font(16, True), fill=(120, 190, 255))
    d.text((1034, 102), "DETECTION", font=font(16, True), fill=(120, 190, 255))
    if dets:
        x1, y1, x2, y2, sc = max(dets, key=lambda k: k[4])
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        x0 = max(0, min(frame.shape[1] - 120, cx - 60))
        y0 = max(0, min(frame.shape[0] - 120, cy - 60))
        crop = cv2.resize(frame[y0:y0 + 120, x0:x0 + 120], (230, 230), interpolation=cv2.INTER_CUBIC)
        canvas.paste(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)), (1034, 130))
        d.text((1034, 366), f"score {sc:.2f}", font=font(18, True), fill=(60, 255, 120))
    d.text((1034, 420), "Green circle =", font=font(15), fill=(200, 210, 230))
    d.text((1034, 440), "detector says", font=font(15), fill=(200, 210, 230))
    d.text((1034, 460), "person. No human", font=font(15), fill=(200, 210, 230))
    d.text((1034, 480), "check of each one.", font=font(15), fill=(200, 210, 230))
    d.text((24, 652), f"{name}  t = {t:.1f} s   detections this frame: {len(dets)}", font=font(22, True), fill=(60, 255, 120) if dets else (255, 150, 150))
    d.text((24, 686), "Cyprus Civil Defence / KIOS search-and-rescue exercise, Zenodo 10.5281/zenodo.5500655, CC BY-SA 4.0. Detector run on 960 px tiles of the 4K frame.", font=font(15), fill=(170, 185, 210))
    return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)


def main():
    det = PersonDetector()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    vw = cv2.VideoWriter(os.path.join(ROOT, "runs", "real", "seg_raw.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (W, H))
    stats = {"detector": "sar_person_yolov8n_v2_best_int8.onnx", "conf": 0.25, "clips": []}
    for fn, label, start in CLIPS:
        cap = cv2.VideoCapture(os.path.join(SRC, fn))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(fps / 10)))
        i, per = 0, []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                dets = detect_tiled(det, frame, conf=0.25)
                per.append(len(dets))
                vw.write(label_frame(frame, dets, label, start + i / fps, len(dets)))
            i += 1
        stats["clips"].append({"file": label, "start_s": start, "frames_analysed": len(per), "frames_with_detection": sum(1 for n in per if n), "mean_detections": sum(per) / max(1, len(per))})
        print(stats["clips"][-1], flush=True)
    vw.release()
    subprocess.check_call(["ffmpeg", "-y", "-loglevel", "error", "-i", os.path.join(ROOT, "runs", "real", "seg_raw.mp4"), "-vf", "format=yuv420p", "-c:v", "libx264", "-crf", "20", OUT])
    with open(STATS, "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
