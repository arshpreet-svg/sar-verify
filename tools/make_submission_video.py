import os
import subprocess
import sys

from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sarverify.overlay import font, wrap

W, H = 1280, 720
BG = (14, 18, 28)
FG = (240, 244, 250)
DIM = (160, 172, 192)
ACC = (90, 160, 255)
WORK = os.path.join(ROOT, "runs", "submission_video")
os.makedirs(WORK, exist_ok=True)
SIM = os.path.join(ROOT, "videos", "sar_verify_demo_seed1.mp4")
OUT = os.path.join(ROOT, "submission", "SAR-Verify_Demo_Video.mp4")


def canvas(title, kicker=None):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 14, H), fill=ACC)
    if kicker:
        d.text((60, 36), kicker.upper(), font=font(20, True), fill=ACC)
    d.text((60, 64), title, font=font(46, True), fill=FG)
    return im, d


def bullets(d, items, y=160, size=29, gap=18, x=60, width=1160, color=FG):
    for it in items:
        lines = wrap(d, it, font(size), width - 30)
        d.ellipse((x, y + 12, x + 9, y + 21), fill=ACC)
        for ln in lines:
            d.text((x + 28, y), ln, font=font(size), fill=color)
            y += size + 8
        y += gap
    return y


def table(d, x, y, cols, rows, widths, size=24, hl=None):
    xs = [x]
    for w in widths:
        xs.append(xs[-1] + w)
    d.rectangle((x - 10, y - 6, xs[-1], y + size + 12), fill=(30, 40, 62))
    for i, c in enumerate(cols):
        d.text((xs[i], y), c, font=font(size, True), fill=ACC)
    y += size + 22
    for r_i, r in enumerate(rows):
        if hl is not None and r_i == hl:
            d.rectangle((x - 10, y - 6, xs[-1], y + size + 10), fill=(26, 60, 44))
        for i, c in enumerate(r):
            d.text((xs[i], y), c, font=font(size, i == 0), fill=FG)
        y += size + 16
    return y


slides = []


def add(name, im, secs):
    p = os.path.join(WORK, name + ".png")
    im.save(p)
    slides.append((p, secs))


im = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(im)
d.rectangle((0, 0, 14, H), fill=ACC)
d.text((60, 220), "SAR-Verify", font=font(96, True), fill=FG)
d.text((60, 350), "Verify before you dispatch", font=font(44), fill=ACC)
d.text((60, 440), "Drones that perceive, decide and act, with a trained person detector in the loop", font=font(28), fill=DIM)
d.text((60, 640), "Physical AI Hackathon  |  Track 5 Disaster Management  |  Simulation: Webots R2025a", font=font(22), fill=DIM)
add("01_title", im, 6)

im, d = canvas("The problem", "1 of 8")
bullets(d, [
    "After an earthquake or flood, drones scan damaged areas faster than ground teams can.",
    "Their person detectors are unreliable: tarps, logs and bags look like people, and people lying in rubble are missed.",
    "Rovers and rescuers are few and slow. Every trip to a false detection delays reaching a real victim.",
    "Ignoring a real detection can cost a life. Trusting everything wastes scarce ground time. Checking everything by hand is slow.",
    "The question: which detections deserve a trip, and when is a second look worth the time?",
], size=29)
add("02_problem", im, 28)

im, d = canvas("Our approach: perceive, decide, act", "2 of 8")
cols = [("PERCEIVE", (40, 130, 255), ["Mothership sweeps at 25 m and flags candidates", "Scout flies close and looks from an angle", "Our trained YOLOv8n person detector runs on the scout camera"]),
        ("DECIDE", (150, 90, 255), ["One belief per candidate, updated by Bayes", "Dynamic program picks: ignore, scout look, ask operator, or dispatch", "Costs are explicit: miss 100, false dispatch 40"]),
        ("ACT", (255, 140, 0), ["Human operator is a hard gate", "Rover drives an A* path around obstacles", "Second mothership pass if victims are still missing"])]
for i, (name, rgb, items) in enumerate(cols):
    x0 = 60 + i * 390
    d.rounded_rectangle((x0, 170, x0 + 370, 640), radius=14, fill=(24, 30, 46), outline=rgb, width=3)
    d.rounded_rectangle((x0, 170, x0 + 370, 222), radius=14, fill=rgb)
    d.text((x0 + 20, 180), f"{i + 1}  {name}", font=font(30, True), fill=(255, 255, 255))
    y = 245
    for it in items:
        for ln in wrap(d, it, font(24), 330):
            d.text((x0 + 20, y), ln, font=font(24), fill=FG)
            y += 32
        y += 14
add("03_approach", im, 24)

im, d = canvas("The simulation", "3 of 8")
bullets(d, [
    "Webots 3D world: buildings, rubble, flood zone, decoys, and people standing or lying in hi-vis clothing.",
    "Three vehicles: mothership drone, scout drone, rover. Third-person camera follows the action.",
    "Labels show what each vehicle perceives, decides and does. The scout camera inset shows the real detector's box.",
    "This run (seed 1): 4 of 4 victims found, 0 wasted rover trips.",
], size=30)
add("04_sim_intro", im, 14)

im, d = canvas("The detector, measured in the simulation", "4 of 8")
d.text((60, 140), "Our YOLOv8n (INT8, trained on VisDrone) run on 15 seeds of rendered scout views", font=font(24), fill=DIM)
table(d, 60, 200, ["Scout sees", "Flagged as person", "95% interval"], [
    ["Standing person", "78 of 84  (93%)", "85 to 97%"],
    ["Person lying, clear", "10 of 51  (20%)", "11 to 33%"],
    ["Person lying, partly covered", "18 of 45  (40%)", "27 to 55%"],
    ["Decoy (tarp, log, bag)", "2 of 270  (0.7%)", "0 to 3%"],
    ["Empty ground", "0 of 450", "0 to 1%"],
], [460, 380, 300], size=28)
d.text((60, 600), "Measured rates feed the AI's beliefs. Straight-down views failed, so the scout looks at an angle.", font=font(24), fill=FG)
d.text((60, 640), "Tested on synthetic renders only. Real-footage performance is not yet measured.", font=font(22), fill=(255, 190, 120))
add("05_detector", im, 22)

im, d = canvas("Results: 200 seeds, 95% intervals", "5 of 8")
chart = Image.open(os.path.join(ROOT, "results", "tradeoff.png")).convert("RGB")
cw = 1160
chart = chart.resize((cw, int(chart.size[1] * cw / chart.size[0])))
im.paste(chart, (60, 130))
y = 130 + chart.size[1] + 14
d.text((60, y), "Left: each point is one false-dispatch cost (10 to 320). Right: realised cost per mission.", font=font(22), fill=DIM)
d.text((60, y + 34), "Trust-first (B0) wastes 2.2 trips per mission. Verification (ours, B2) wastes 0.2.", font=font(26, True), fill=FG)
add("06_chart", im, 26)

im, d = canvas("What the numbers say", "6 of 8")
d.text((60, 140), "False-dispatch cost 40, 200 seeds", font=font(24), fill=DIM)
table(d, 60, 185, ["Policy", "Victims found", "Wasted trips", "Time to confirm"], [
    ["B0 trust first", "0.79", "2.20", "409 s"],
    ["B1 tuned threshold", "0.68", "1.05", "401 s"],
    ["B2 always verify", "0.79", "0.20", "426 s"],
    ["Ours (DP policy)", "0.79", "0.20", "426 s"],
], [430, 260, 260, 260], size=28, hl=3)
bullets(d, [
    "Same recall as trust-first with about 91% fewer wasted trips, at about 4% slower confirmation.",
    "Ours equals always-verify when asking is cheap. It beats trust-first once a false dispatch costs 40 or more.",
    "When asking the operator is costly, ours asks less (2.3 against 5.5 questions) and saves 18 to 33 cost units.",
    "A second mothership pass raised recall from 0.68 to 0.79.",
], y=440, size=24, gap=8)
add("07_table", im, 34)

im, d = canvas("The path to reality", "7 of 8")
table(d, 60, 150, ["Part", "In simulation", "On hardware (plan)"], [
    ["Scout perception", "INT8 YOLOv8n, ONNX Runtime", "Same model on a Raspberry Pi 5 + gimbal camera"],
    ["Decision", "Pure Python, 1.7 microseconds each", "Runs unchanged on the Pi 5"],
    ["Aircraft", "Simplified quadrotor", "Off-the-shelf quadrotor"],
    ["Rover", "A* path planner", "Small ground robot with GNSS, same path"],
    ["Operator", "Delay and error model", "Tablet: scout image, yes/no button"],
], [260, 430, 530], size=22)
d.text((60, 560), "The decision code has no simulator dependency, so the same code goes onto the hardware.", font=font(24, True), fill=FG)
d.text((60, 600), "Measured: detector 24 ms per frame on a laptop CPU. Not yet measured: Raspberry Pi 5 speed, real footage.", font=font(22), fill=(255, 190, 120))
d.text((60, 636), "Next: real aerial images, Pi 5 timing, recorded-video tests, then a low-altitude flight with a mannequin.", font=font(22), fill=DIM)
add("08_reality", im, 32)

im, d = canvas("Honest limits", "8 of 8")
bullets(d, [
    "The detector was tested only on synthetic renders. Real-world rates are unknown.",
    "The mothership's wide scan still uses assumed detection rates.",
    "People lying down are found only 20 to 40% of the time. The operator gate and the second pass cover for this.",
    "Timings come from a fast simulator matched to one Webots run. No hardware yet.",
], size=30)
d.text((60, 600), "SAR-Verify: verify before you dispatch", font=font(34, True), fill=ACC)
add("09_limits", im, 20)

parts = []
for i, (p, secs) in enumerate(slides):
    out = os.path.join(WORK, f"seg{i:02d}.mp4")
    subprocess.check_call(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-t", str(secs), "-i", p, "-vf", "scale=1280:720,format=yuv420p", "-r", "25", "-c:v", "libx264", "-preset", "medium", "-crf", "20", out])
    parts.append(out)
simseg = os.path.join(WORK, "sim.mp4")
subprocess.check_call(["ffmpeg", "-y", "-loglevel", "error", "-i", SIM, "-vf", "scale=1280:720,format=yuv420p", "-r", "25", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-an", simseg])
order = parts[:4] + [simseg] + parts[4:]
lst = os.path.join(WORK, "list.txt")
with open(lst, "w") as f:
    for p in order:
        f.write(f"file '{p}'\n")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
subprocess.check_call(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", OUT])
print(OUT)
