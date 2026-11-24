import math

VIEW_PITCH = 0.75
OVERVIEW = (0, -92, 88)


def beacon(rgb, z, r):
    return (
        f"Pose {{ translation 0 0 {z} children [ Shape {{ appearance Appearance {{ material Material "
        f"{{ diffuseColor 0 0 0 emissiveColor {rgb[0]} {rgb[1]} {rgb[2]} }} }} geometry Sphere {{ radius {r} }} }} ] }}"
    )


PROTO_BASE = "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects"

EXTERN = [
    "objects/backgrounds/protos/TexturedBackground.proto",
    "objects/backgrounds/protos/TexturedBackgroundLight.proto",
    "objects/floors/protos/Floor.proto",
    "appearances/protos/SandyGround.proto",
    "robots/dji/mavic/protos/Mavic2Pro.proto",
    "robots/adept/pioneer3/protos/Pioneer3at.proto",
    "humans/pedestrian/protos/Pedestrian.proto",
]


def _qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def look_orientation(yaw, pitch_down):
    qz = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
    qy = (0.0, math.sin(pitch_down / 2), 0.0, math.cos(pitch_down / 2))
    x, y, z, w = _qmul(qz, qy)
    angle = 2 * math.acos(max(-1.0, min(1.0, w)))
    s = math.sqrt(max(1e-12, 1 - w * w))
    return x / s, y / s, z / s, angle


def _box(name, x, y, z, sx, sy, sz, yaw, rgb, extra=""):
    return f"""Solid {{
  translation {x:.3f} {y:.3f} {z:.3f}
  rotation 0 0 1 {yaw:.4f}
  name "{name}"
  children [
    Shape {{
      appearance PBRAppearance {{ baseColor {rgb[0]} {rgb[1]} {rgb[2]} roughness 1 metalness 0 {extra} }}
      geometry Box {{ size {sx:.3f} {sy:.3f} {sz:.3f} }}
    }}
  ]
}}
"""


SHIRTS = [(1.0, 0.45, 0.0), (1.0, 0.92, 0.0), (1.0, 0.1, 0.1), (0.1, 0.85, 1.0), (1.0, 0.4, 0.7)]
PANTS = [(0.15, 0.2, 0.45), (0.25, 0.22, 0.2), (0.1, 0.1, 0.12), (0.3, 0.3, 0.5)]


def _person(name, x, y, yaw, idx=0, pose="lying"):
    sh = SHIRTS[idx % len(SHIRTS)]
    pa = PANTS[idx % len(PANTS)]
    return f"""Pose {{
  translation {x:.3f} {y:.3f} 0
  rotation 0 0 1 {yaw:.4f}
  children [
    Pedestrian {{
      translation 0 0 {0.22 if pose == "lying" else 1.27}
      rotation 0 1 0 {1.5708 if pose == "lying" else 0}
      name "{name}"
      controller "<none>"
      shirtColor {sh[0]} {sh[1]} {sh[2]}
      pantsColor {pa[0]} {pa[1]} {pa[2]}
    }}
  ]
}}
"""


def _decoy(name, x, y, yaw, kind):
    if kind == "tarp":
        return _box(name, x, y, 0.05, 1.7, 0.8, 0.1, yaw, (0.2, 0.35, 0.8))
    if kind == "log":
        return _box(name, x, y, 0.14, 1.8, 0.28, 0.28, yaw, (0.35, 0.22, 0.1))
    return _box(name, x, y, 0.15, 1.0, 0.55, 0.3, yaw, (0.12, 0.12, 0.14))


def _building(i, o):
    s = _box(f"building_{i}", o["x"], o["y"], o["h"] / 2, o["sx"], o["sy"], o["h"], o["yaw"], (0.55, 0.53, 0.5))
    s += _box(
        f"roof_{i}", o["x"] + 0.5, o["y"] - 0.5, o["h"] + 0.25, o["sx"] * 0.7, o["sy"] * 0.5, 0.4,
        o["yaw"] + 0.4, (0.4, 0.2, 0.15),
    )
    return s


def build_world(scene, cfg, calib=False, pose="lying"):
    lines = ["#VRML_SIM R2025a utf8", ""]
    for e in EXTERN:
        lines.append(f'EXTERNPROTO "{PROTO_BASE}/{e}"')
    lines.append('IMPORTABLE EXTERNPROTO "../protos/Marker.proto"')
    lines.append("")
    lines.append("WorldInfo {")
    lines.append('  title "SAR-Verify"')
    lines.append("  basicTimeStep 8")
    lines.append("  defaultDamping Damping { linear 0.5 angular 0.5 }")
    lines.append("}")
    ox, oy, oz, oa = look_orientation(math.pi / 2, VIEW_PITCH)
    lines.append(
        f"DEF VIEW Viewpoint {{\n  orientation {ox:.5f} {oy:.5f} {oz:.5f} {oa:.5f}\n  position {OVERVIEW[0]} {OVERVIEW[1]} {OVERVIEW[2]}\n  near 0.5\n}}"
    )
    lines.append("TexturedBackground { luminosity 1.2 }")
    lines.append("TexturedBackgroundLight { }")
    lines.append("Floor {\n  size 220 220\n  tileSize 8 8\n  appearance SandyGround { }\n}")
    for i, o in enumerate([o for o in scene["obstacles"] if o["kind"] == "building"]):
        lines.append(_building(i, o))
    for i, o in enumerate([o for o in scene["obstacles"] if o["kind"] == "rubble"]):
        lines.append(_box(f"rubble_{i}", o["x"], o["y"], o["h"] / 2, o["sx"], o["sy"], o["h"], o["yaw"], (0.42, 0.36, 0.3)))
    for o in [o for o in scene["obstacles"] if o["kind"] == "flood"]:
        lines.append(
            _box("flood", o["x"], o["y"], 0.03, o["sx"], o["sy"], 0.06, 0.0, (0.1, 0.35, 0.6), "transparency 0.35")
        )
    for v in scene["victims"]:
        lines.append(f"DEF VICTIM_{v['id']} " + _person(f"victim_{v['id']}", v["x"], v["y"], v["yaw"], v["id"] if isinstance(v["id"], int) else 0, v.get("pose", pose)))
        if v["occlusion"] > 0:
            cw = 0.6 + 1.4 * v["occlusion"]
            lines.append(_box(f"cover_{v['id']}", v["x"], v["y"], 0.3, cw, 1.0, 0.1, v["yaw"], (0.4, 0.36, 0.3)))
    for d in scene["decoys"]:
        lines.append(f"DEF DECOY_{d['id']} " + _decoy(f"decoy_{d['id']}", d["x"], d["y"], d["yaw"], d["kind"]))
    st = scene["start"]
    radio = "Emitter { }\n      Receiver { }"
    cam = "Camera { width 640 height 480 fieldOfView 1.0 near 0.2 antiAliasing TRUE }"
    lines.append(
        f"""DEF MOTHERSHIP Mavic2Pro {{
  translation {st['mothership'][0]:.2f} {st['mothership'][1]:.2f} 0.1
  name "mothership"
  controller "mothership"
  bodySlot [
      {radio}
      {beacon((0.2, 0.4, 1.0), 0.5, 0.55)}
  ]
  cameraSlot [ {cam} ]
}}"""
    )
    lines.append(
        f"""DEF SCOUT Mavic2Pro {{
  translation {st['scout'][0]:.2f} {st['scout'][1]:.2f} 0.1
  name "scout"
  controller "scout"
  bodySlot [
      {radio}
      {beacon((1.0, 0.2, 0.85), 0.45, 0.45)}
  ]
  cameraSlot [ {cam} ]
}}"""
    )
    lines.append(
        f"""DEF ROVER Pioneer3at {{
  translation {st['rover'][0]:.2f} {st['rover'][1]:.2f} 0.15
  name "rover"
  controller "rover"
  extensionSlot [
      GPS {{ }}
      Compass {{ }}
      {radio}
      {beacon((1.0, 1.0, 1.0), 0.75, 0.3)}
      Camera {{ translation 0.3 0 0.2 width 160 height 120 }}
  ]
}}"""
    )
    lines.append(
        f"""DEF COMMANDER Robot {{
  name "commander"
  controller "commander"
  children [
      {radio}
  ]
}}"""
    )
    lines.append(
        f"""DEF REFEREE Robot {{
  name "referee"
  controller "referee"
  supervisor TRUE
  children [
      {radio}
      Camera {{ name "cine" width 1280 height 720 fieldOfView 1.15 near 0.3 antiAliasing TRUE }}
  ]
}}"""
    )
    if calib:
        lines = [l for l in lines if not l.startswith(("DEF MOTHERSHIP", "DEF SCOUT", "DEF ROVER", "DEF COMMANDER", "DEF REFEREE"))]
        lines.append(
            'Robot {\n  name "calibrator"\n  controller "calibrator"\n  supervisor TRUE\n  children [\n'
            '      Camera { name "cam" width 640 height 480 fieldOfView 1.0 near 0.2 antiAliasing TRUE }\n  ]\n}'
        )
    return "\n".join(lines) + "\n"


def write_world(scene, cfg, path, calib=False, pose="lying"):
    with open(path, "w") as f:
        f.write(build_world(scene, cfg, calib, pose))
