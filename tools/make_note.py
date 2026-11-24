import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "submission", "SAR-Verify_Concept_Note.pdf")

ss = getSampleStyleSheet()
body = ParagraphStyle("b", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.4, leading=12.4, alignment=TA_LEFT, spaceAfter=4)
h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=17, leading=20, spaceAfter=2)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, leading=14, spaceBefore=7, spaceAfter=3, keepWithNext=1, textColor=colors.HexColor("#1a3a6b"))
sub = ParagraphStyle("s", parent=body, textColor=colors.HexColor("#444444"), fontSize=9)
cell = ParagraphStyle("c", parent=body, fontSize=8.4, leading=10.4, spaceAfter=0)
cellb = ParagraphStyle("cb", parent=cell, fontName="Helvetica-Bold")


def P(t, s=body):
    return Paragraph(t, s)


def table(rows, widths, header=True):
    data = [[P(c, cellb if (header and i == 0) else cell) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4), ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef8")))
    t.setStyle(TableStyle(st))
    return t


doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=15 * mm, bottomMargin=14 * mm, title="SAR-Verify concept note")
W = A4[0] - 36 * mm
s = []
s.append(P("SAR-Verify: verify before you dispatch", h1))
s.append(P("Physical AI Hackathon, Track 5 Disaster Management (Robotics + AI). Concept note for Stage 1.", sub))
s.append(P("Team: [TEAM NAME AND MEMBERS TO BE FILLED]. Code: https://github.com/arshpreet-svg/sar-verify. Simulator: Webots R2025a.", sub))

s.append(P("1. Problem", h2))
s.append(P("After an earthquake or flood, drones can cover a damaged area far faster than ground teams, and a person detector on the drone flags possible victims. The detections are unreliable: debris, tarps and bags look like people, and people lying in rubble are missed. Ground rovers and rescuers are few and slow, so every trip to a false detection delays reaching a real victim, while ignoring a real detection can cost a life. Trusting every detection wastes scarce ground time. Verifying every detection by hand is slow and ties up the human operator. The beneficiaries are search-and-rescue teams and the people trapped in the first hours. Which detections deserve a trip, and when to spend time on a second look, is the decision this project automates.", body))

s.append(P("2. Approach", h2))
s.append(P("SAR-Verify is a three-vehicle team that perceives, decides and acts. A <b>mothership</b> drone sweeps the area at 25 m and flags candidates. A <b>scout</b> drone flies to a candidate, looks from an angle at 12 m altitude and 14 m standoff, and runs our trained YOLOv8n person detector on its own camera image. A <b>rover</b> drives an A* path around obstacles to a confirmed candidate. A <b>human operator</b> is a hard gate: no rover leaves without an operator yes. A <b>commander</b> holds one Bayesian belief per candidate and picks the next action with a finite-horizon dynamic program over four actions: ignore, scout look, ask the operator, dispatch the rover. The costs are explicit and adjustable: miss 100, false dispatch 40, travel 5, look 8, ask 4, at most two looks per candidate.", body))
s.append(table([
    ["Stage", "What happens", "Where"],
    ["Perceive", "Mothership flags candidates (sensor model, assumed rates). Scout runs the real INT8 YOLOv8n detector on rendered camera frames; a hit counts if a detection lies within 3 m of the projected candidate.", "sarverify/sensor.py, detector.py, controllers/scout"],
    ["Decide", "Bayes update of each candidate's belief; dynamic program chooses ignore / look / ask / dispatch; operator model with delay and error rates.", "sarverify/belief.py, policy.py, operator.py"],
    ["Act", "Scout stand-off look, operator confirmation gate, rover A* dispatch, second mothership pass if victims remain unaccounted for.", "controllers/*, sarverify/planner.py"],
], [20 * mm, 104 * mm, W - 124 * mm]))
s.append(P("The decision code in <font face='Courier'>sarverify/</font> is pure Python with no simulator import, so the same code runs in the 3D simulation, in the batch evaluator and on embedded hardware. The simulation shows people (standing and lying, in hi-vis clothing), decoys, buildings, rubble and a flooded zone, with a third-person camera and an overlay that labels what each vehicle perceives, decides and does.", body))

s.append(P("3. Detector measured in the loop", h2))
s.append(P("We rendered the scout's camera view in Webots for 15 calibration seeds (separate from evaluation seeds) and ran the detector on every frame. The measured rates feed the belief update, and the fast evaluator samples scout looks from them. Straight-down views performed poorly for every pose, which is why the scout looks obliquely.", body))
s.append(table([
    ["Target seen by the scout", "Flagged as person", "95% interval"],
    ["Standing person", "78 of 84 (93%)", "85 to 97%"],
    ["Person lying, unobstructed", "10 of 51 (20%)", "11 to 33%"],
    ["Person lying, partly covered", "18 of 45 (40%)", "27 to 55%"],
    ["Decoy (tarp, log, bag)", "2 of 270 (0.7%)", "0 to 3%"],
    ["Empty ground", "0 of 450", "0 to 1%"],
], [70 * mm, 50 * mm, W - 120 * mm]))
s.append(P("The detector was trained on real aerial footage (VisDrone) but tested here only on synthetic renders, so real-world rates are unmeasured. As a qualitative check, we ran the detector on two 6 s clips of public 4K drone footage from a real search-and-rescue exercise (Cyprus Civil Defence and KIOS, CC BY-SA 4.0), using 960 px tiles. It flagged people on a roof in some frames and missed others, and likely false alarms also appeared. We have not hand-labelled these frames, so no accuracy figure is claimed. The mothership's wide scan still samples from assumed detection curves.", body))

s.append(P("4. Simulation results", h2))
s.append(P("A fast pure-Python mission model reuses the same policy, belief, operator, scene and path-planning code. Its speeds were matched to one Webots run, so timings are approximate. We compared four policies on 200 evaluation seeds, with the B1 threshold tuned on 60 separate seeds, and report 95% bootstrap intervals. B0 trusts the first detection. B1 asks the operator above a tuned confidence threshold. B2 always looks once, then asks. Ours is the dynamic-program policy. Values below are at false-dispatch cost 40.", body))
s.append(table([
    ["Policy", "Victims found", "Wasted rover trips", "Mean time to confirm", "Realised cost"],
    ["B0 trust first", "0.79 [0.76, 0.81]", "2.20 [2.04, 2.36]", "409 s [394, 425]", "200 [187, 213]"],
    ["B1 tuned threshold", "0.68 [0.65, 0.71]", "1.05 [0.93, 1.19]", "401 s [385, 417]", "215 [201, 229]"],
    ["B2 always verify", "0.79 [0.76, 0.81]", "0.20 [0.15, 0.26]", "426 s [412, 441]", "181 [171, 192]"],
    ["Ours (DP)", "0.79 [0.76, 0.81]", "0.20 [0.15, 0.26]", "426 s [412, 441]", "181 [171, 192]"],
], [34 * mm, 34 * mm, 34 * mm, 38 * mm, W - 140 * mm]))
s.append(Spacer(1, 3))
img = Image(os.path.join(ROOT, "results", "tradeoff.png"), width=W, height=W * 4.6 / 12)
s.append(img)
s.append(P("Findings, as measured:", body))
s.append(P("&bull; Verification gives the same recall as trust-first with about 91% fewer wasted trips (0.20 against 2.20 per mission), at about 17 s (4%) slower mean time to confirm.<br/>"
           "&bull; Ours matches B2 exactly when asking the operator is cheap. Realised cost is lower than B0 once a false dispatch costs 40 or more, by 99 at cost 80 and 259 at cost 160 (intervals exclude zero). At cost 10 trust-first is cheaper.<br/>"
           "&bull; Operator scarcity sweep (4 to 40 per question, 6 to 120 s answer time, one answer at a time): ours separates from B2 only when a question costs as much as a false dispatch (40), where it asks about 2.3 times per mission against 5.5 and saves 18 to 33 cost units. When asking is expensive and false dispatches are cheap, B0 is cheaper than ours.<br/>"
           "&bull; A second mothership pass when victims remain unaccounted for raised recall from 0.68 to 0.79. In Webots, seed 3 went from 1 of 4 to 3 of 4 victims.", body))

s.append(P("5. Path to reality", h2))
s.append(table([
    ["Element", "Simulation", "Hardware plan", "Status"],
    ["Perception on the scout", "ONNX Runtime, INT8 YOLOv8n, 960 px", "Same model on a Raspberry Pi 5 with a camera on a gimbal at 40 degrees down", "Model package already targets Pi 5. Inference is 24 ms on a laptop-class CPU (Apple M4 Pro); Pi 5 latency not yet measured."],
    ["Decision", "Pure Python DP; solve takes 1.5 ms, one decision 1.7 microseconds on the laptop", "Runs on the Pi 5 with no change", "Latency on the Pi to be measured"],
    ["Aircraft", "Simplified quadrotor dynamics", "Off-the-shelf quadrotor with a flight controller accepting position setpoints; two units", "Not built"],
    ["Rover", "A* on a 1 m grid, 1.5 m clearance", "Small ground robot with GNSS and obstacle sensing following the planned path", "Not built"],
    ["Operator", "Delay and error model", "Tablet showing the scout image with a yes/no button", "Not built"],
    ["Radio", "Webots emitter and receiver", "Message bus between vehicles and commander, same JSON messages", "Not built"],
], [28 * mm, 42 * mm, 56 * mm, W - 126 * mm]))
s.append(P("Steps before a field test: measure the detector on real aerial images and replace the assumed mothership curves with measured ones; re-estimate the scout rates on real footage; measure Pi 5 latency; run the pipeline with recorded flight video; then a tethered or low-altitude flight with a decoy and a mannequin.", body))

s.append(P("6. Expected impact and limits", h2))
s.append(P("The expected impact is fewer wasted ground trips at the same coverage, so ground teams reach real victims sooner. The result is credible in simulation but its size depends on assumptions we state openly: the false-dispatch and ask costs, the operator model, the assumed mothership detector, and a detector tested only on synthetic renders. Detector recall on people lying down is low (20 to 40%), so the operator gate and the second pass matter. The evaluation does not include mission time in the cost. We have not run hardware.", body))
s.append(P("Real footage: KIOS Research and Innovation Center of Excellence, Drones' footage of a search-and-rescue exercise scenario, Zenodo 10.5281/zenodo.5500655, CC BY-SA 4.0. Open-source components: Webots, ONNX Runtime, OpenCV, Ultralytics YOLOv8 and the VisDrone dataset (used for the detector's training), NumPy, Pillow, ReportLab.", sub))
doc.build(s)
