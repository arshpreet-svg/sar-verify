from PIL import Image, ImageDraw, ImageFont

from .cinema import project

FONT_PATHS = ["/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/SFNS.ttf"]

STATUS_NAME = {
    "new": "NEW",
    "wait_scout": "NEEDS SCOUT",
    "scouting": "SCOUT LOOKING",
    "wait_operator": "ASK OPERATOR",
    "wait_rover": "DISPATCHING",
    "rover_going": "ROVER GOING",
    "confirmed": "CONFIRMED",
    "rejected": "REJECTED",
    "wasted": "FALSE ALARM",
}
STATUS_RGB = {
    "new": (255, 217, 0),
    "wait_scout": (255, 217, 0),
    "scouting": (51, 179, 255),
    "wait_operator": (190, 140, 255),
    "wait_rover": (255, 140, 0),
    "rover_going": (255, 140, 0),
    "confirmed": (26, 230, 51),
    "rejected": (150, 150, 150),
    "wasted": (242, 38, 38),
}
PHASE_RGB = {"PERCEIVE": (40, 130, 255), "DECIDE": (150, 90, 255), "ACT": (255, 140, 0)}
VEHICLE_RGB = {"MOTHERSHIP": (60, 110, 255), "SCOUT": (255, 60, 210), "ROVER": (240, 240, 240)}

_fonts = {}


def font(size, bold=False):
    key = (size, bold)
    if key not in _fonts:
        f = None
        for p in FONT_PATHS:
            try:
                f = ImageFont.truetype(p, size, index=1 if bold and p.endswith(".ttc") else 0)
                break
            except Exception:
                continue
        _fonts[key] = f or ImageFont.load_default()
    return _fonts[key]


def wrap(draw, text, fnt, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=fnt) <= width:
            cur = t
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def pill(draw, xy, text, rgb, size=15, anchor="mb", fg=(255, 255, 255), alpha=215):
    f = font(size, True)
    w = draw.textlength(text, font=f)
    x, y = xy
    pad = 6
    h = size + 6
    if anchor == "mb":
        box = (x - w / 2 - pad, y - h, x + w / 2 + pad, y)
    else:
        box = (x, y, x + w + 2 * pad, y + h)
    draw.rounded_rectangle(box, radius=7, fill=rgb + (alpha,))
    draw.text((box[0] + pad, box[1] + 2), text, font=f, fill=fg)


def render(path, out, s):
    img = Image.open(path).convert("RGBA")
    W, H = img.size
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    cam, tgt, hfov = s["cam"], s["target"], s["hfov"]

    def proj(p):
        r = project(p, cam, tgt, W, H, hfov)
        if r is None or not (0 <= r[0] <= W and 0 <= r[1] <= H):
            return None
        return r

    for t in s["truth"]:
        r = proj((t["x"], t["y"], 1.0))
        if r is None or r[2] > 60:
            continue
        if t["kind"] == "victim":
            rad = max(16.0, 2.2 * 1170.0 / r[2] / 2.0)
            d.ellipse((r[0] - rad, r[1] - rad, r[0] + rad, r[1] + rad), outline=(255, 255, 255, 235), width=3)
            pill(d, (r[0], r[1] - rad - 2), "HUMAN (" + t.get("pose", "") + ")", (20, 20, 20), 13, alpha=190)
        else:
            pill(d, (r[0], r[1] - 8), "decoy: " + t["name"], (70, 70, 70), 12, alpha=150)
    for name, p in s["vehicles"].items():
        r = proj((p[0], p[1], p[2] + 1.6))
        if r is None:
            continue
        pill(d, (r[0], r[1] - 6), name, VEHICLE_RGB[name], 17)
        d.polygon([(r[0] - 7, r[1] - 6), (r[0] + 7, r[1] - 6), (r[0], r[1] + 4)], fill=VEHICLE_RGB[name] + (230,))
    for c in s["cands"]:
        r = proj((c["pos"][0], c["pos"][1], 3.7))
        if r is None:
            continue
        rgb = STATUS_RGB.get(c["status"], (255, 255, 255))
        txt = f"#{c['id']}  P(person)={c['belief']:.2f}  {STATUS_NAME.get(c['status'], c['status'])}"
        pill(d, (r[0], r[1]), txt, rgb, 15, fg=(0, 0, 0) if c["status"] not in ("wasted", "scouting", "wait_operator") else (0, 0, 0))

    d.rectangle((0, 0, W, 54), fill=(12, 16, 24, 225))
    d.text((18, 9), "SAR-VERIFY", font=font(30, True), fill=(255, 255, 255))
    d.text((190, 17), "autonomous search & rescue: perceive  >  decide  >  act", font=font(19), fill=(170, 185, 210))
    score = f"VICTIMS FOUND {s['found']}/{s['n_victims']}    WASTED TRIPS {s['wasted']}"
    d.text((W - 18 - d.textlength(score, font=font(22, True)), 14), score, font=font(22, True), fill=(120, 255, 150))
    clock = f"t = {int(s['t']) // 60}:{int(s['t']) % 60:02d}   x{s['speed']:.0f}"
    d.text((W - 18 - d.textlength(clock, font=font(18)), 60), clock, font=font(18, True), fill=(255, 255, 255))

    px, pw, gap = 18, 300, 10
    for i, name in enumerate(("PERCEIVE", "DECIDE", "ACT")):
        text, age = s["phases"].get(name, ("", 1e9))
        x0 = px + i * (pw + gap)
        rgb = PHASE_RGB[name]
        hot = age < 6.0
        d.rounded_rectangle((x0, 62, x0 + pw, 150), radius=9, fill=(12, 16, 24, 220), outline=rgb + ((255 if hot else 90),), width=3 if hot else 1)
        d.rounded_rectangle((x0, 62, x0 + pw, 86), radius=9, fill=rgb + ((255 if hot else 120),))
        d.text((x0 + 10, 64), f"{i + 1}  {name}", font=font(17, True), fill=(255, 255, 255))
        lines = wrap(d, text or "waiting", font(15), pw - 18)[:3]
        for j, ln in enumerate(lines):
            d.text((x0 + 10, 90 + j * 19), ln, font=font(15), fill=(255, 255, 255) if hot else (150, 160, 175))

    cs = s["cands"]
    if cs:
        x0, y0 = W - 300, 90
        d.rounded_rectangle((x0, y0, W - 12, y0 + 30 + 30 * len(cs)), radius=9, fill=(12, 16, 24, 215))
        d.text((x0 + 10, y0 + 6), "CANDIDATES  (AI belief)", font=font(15, True), fill=(200, 210, 230))
        for i, c in enumerate(cs):
            y = y0 + 32 + 30 * i
            rgb = STATUS_RGB.get(c["status"], (255, 255, 255))
            d.text((x0 + 10, y + 3), f"#{c['id']}", font=font(16, True), fill=(255, 255, 255))
            d.rectangle((x0 + 48, y + 6, x0 + 118, y + 20), outline=(120, 130, 150, 255))
            d.rectangle((x0 + 48, y + 6, x0 + 48 + 70 * c["belief"], y + 20), fill=rgb + (255,))
            d.text((x0 + 126, y + 3), STATUS_NAME.get(c["status"], c["status"]), font=font(14, True), fill=rgb + (255,))

    items = [("MOTHERSHIP drone (wide scan)", VEHICLE_RGB["MOTHERSHIP"]), ("SCOUT drone (close look)", VEHICLE_RGB["SCOUT"]), ("ROVER (ground)", VEHICLE_RGB["ROVER"]), ("candidate marker = AI suspects a person", STATUS_RGB["new"])]
    y0 = H - 18 - 24 * len(items)
    d.rounded_rectangle((12, y0 - 10, 330, H - 8), radius=9, fill=(12, 16, 24, 200))
    for i, (txt, rgb) in enumerate(items):
        yy = y0 + 24 * i
        d.ellipse((22, yy + 3, 36, yy + 17), fill=rgb + (255,))
        d.text((44, yy), txt, font=font(14), fill=(255, 255, 255))

    ins = s.get("inset")
    if ins:
        try:
            im = Image.open(ins["img"]).convert("RGBA")
            iw, ih = 400, 300
            sx, sy = iw / im.size[0], ih / im.size[1]
            im = im.resize((iw, ih))
            dd = ImageDraw.Draw(im)
            for x1, y1, x2, y2, sc in ins["dets"]:
                dd.rectangle((x1 * sx, y1 * sy, x2 * sx, y2 * sy), outline=(0, 255, 60, 255), width=3)
                dd.text((x1 * sx, max(0, y1 * sy - 16)), f"person {sc:.2f}", font=font(14, True), fill=(0, 255, 60, 255))
            if ins.get("target_px"):
                tx, ty = ins["target_px"][0] * sx, ins["target_px"][1] * sy
                dd.ellipse((tx - 12, ty - 12, tx + 12, ty + 12), outline=(255, 60, 60, 255), width=2)
            x0, y0 = W - iw - 14, H - ih - 100
            d.rounded_rectangle((x0 - 6, y0 - 34, x0 + iw + 6, y0 + ih + 40), radius=10, fill=(12, 16, 24, 235))
            d.text((x0 + 4, y0 - 28), "SCOUT CAMERA + AI PERSON DETECTOR", font=font(14, True), fill=(255, 60, 210, 255))
            layer.paste(im, (int(x0), int(y0)))
            res = f"#{ins['cand']}: PERSON {ins['score']:.2f}" if ins["flag"] else f"#{ins['cand']}: no person"
            d.text((x0 + 4, y0 + ih + 8), res, font=font(17, True), fill=(60, 255, 120, 255) if ins["flag"] else (255, 120, 120, 255))
            d.text((x0 + iw - 128, y0 + ih + 12), "red ring = candidate", font=font(12), fill=(255, 150, 150, 255))
        except Exception:
            pass

    if s.get("banner"):
        text, rgb = s["banner"]
        f = font(40, True)
        w = d.textlength(text, font=f)
        d.rounded_rectangle((W / 2 - w / 2 - 30, H - 120, W / 2 + w / 2 + 30, H - 52), radius=14, fill=rgb + (235,))
        d.text((W / 2 - w / 2, H - 112), text, font=f, fill=(255, 255, 255))

    Image.alpha_composite(img, layer).convert("RGB").save(out, quality=90)
