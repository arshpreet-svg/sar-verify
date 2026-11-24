import math

K_VERTICAL_THRUST = 68.5
K_VERTICAL_OFFSET = 0.6
K_VERTICAL_P = 3.0
K_ROLL_P = 50.0
K_PITCH_P = 30.0
PITCH_RATE = 2.0
ROLL_RATE = 4.0


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def wrap(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


class Quad:
    def __init__(self, robot, step, max_speed=5.0, yaw_target=0.0):
        self.robot = robot
        self.step = step
        self.max_speed = max_speed
        self.yaw_target = yaw_target
        self.camera = robot.getDevice("camera")
        self.camera.enable(step * 4)
        self.imu = robot.getDevice("inertial unit")
        self.imu.enable(step)
        self.gps = robot.getDevice("gps")
        self.gps.enable(step)
        self.compass = robot.getDevice("compass")
        self.compass.enable(step)
        self.gyro = robot.getDevice("gyro")
        self.gyro.enable(step)
        self.cam_roll = robot.getDevice("camera roll")
        self.cam_pitch = robot.getDevice("camera pitch")
        names = ["front left propeller", "front right propeller", "rear left propeller", "rear right propeller"]
        self.motors = [robot.getDevice(n) for n in names]
        for m in self.motors:
            m.setPosition(float("inf"))
            m.setVelocity(1.0)
        self.led_l = robot.getDevice("front left led")
        self.led_r = robot.getDevice("front right led")
        self.target_alt = 1.0
        self.cam_pitch_cmd = 1.5
        self.target_xy = None
        self.v_des = (0.0, 0.0)
        self.pd = 0.0
        self.rd = 0.0
        self.dt = step / 1000.0

    def position(self):
        v = self.gps.getValues()
        return v[0], v[1], v[2]

    def velocity(self):
        v = self.gps.getSpeedVector()
        return v[0], v[1], v[2]

    def yaw(self):
        return self.imu.getRollPitchYaw()[2]

    def goto(self, x, y, alt=None):
        self.target_xy = (x, y)
        if alt is not None:
            self.target_alt = alt

    def hold(self):
        px, py, _ = self.position()
        self.target_xy = (px, py)

    def distance_to_target(self):
        if self.target_xy is None:
            return 0.0
        px, py, _ = self.position()
        return math.hypot(self.target_xy[0] - px, self.target_xy[1] - py)

    def at_target(self, tol_xy=1.5, tol_alt=1.0, max_speed=0.8):
        if self.target_xy is None:
            return False
        _, _, pz = self.position()
        vx, vy, _ = self.velocity()
        return (
            self.distance_to_target() < tol_xy
            and abs(pz - self.target_alt) < tol_alt
            and math.hypot(vx, vy) < max_speed
        )

    def update(self):
        t = self.robot.getTime()
        roll, pitch, yaw = self.imu.getRollPitchYaw()
        px, py, pz = self.position()
        gx, gy, _ = self.gyro.getValues()
        vx, vy, _ = self.velocity()
        on = int(t) % 2
        self.led_l.set(on)
        self.led_r.set(1 - on)
        self.cam_roll.setPosition(clamp(-0.115 * gx, -0.4, 0.4))
        self.cam_pitch.setPosition(clamp(self.cam_pitch_cmd - 0.1 * gy, -0.4, 1.65))

        roll_dist = 0.0
        pitch_dist = 0.0
        if self.target_xy is not None and pz > 0.6 * max(self.target_alt, 1.0):
            dx = self.target_xy[0] - px
            dy = self.target_xy[1] - py
            dist = math.hypot(dx, dy)
            speed = min(self.max_speed, 0.45 * dist)
            if dist > 1e-6:
                wvx = speed * dx / dist
                wvy = speed * dy / dist
            else:
                wvx = wvy = 0.0
            cy, sy = math.cos(yaw), math.sin(yaw)
            des_f = cy * wvx + sy * wvy
            des_l = -sy * wvx + cy * wvy
            cur_f = cy * vx + sy * vy
            cur_l = -sy * vx + cy * vy
            pitch_dist = clamp(-(1.5 * des_f + 0.3 * (des_f - cur_f)), -3.5, 3.5)
            roll_dist = clamp(2.5 * des_l + 0.5 * (des_l - cur_l), -7.0, 7.0)
            self.v_des = (wvx, wvy)

        self.pd += clamp(pitch_dist - self.pd, -PITCH_RATE * self.dt, PITCH_RATE * self.dt)
        self.rd += clamp(roll_dist - self.rd, -ROLL_RATE * self.dt, ROLL_RATE * self.dt)
        pitch_dist = self.pd
        roll_dist = self.rd
        yaw_dist = clamp(1.5 * wrap(self.yaw_target - yaw), -1.3, 1.3)
        roll_in = K_ROLL_P * clamp(roll, -1.0, 1.0) + gx + roll_dist
        pitch_in = K_PITCH_P * clamp(pitch, -1.0, 1.0) + gy + pitch_dist
        diff = clamp(self.target_alt - pz + K_VERTICAL_OFFSET, -1.0, 1.0)
        vert = K_VERTICAL_P * diff ** 3
        fl = K_VERTICAL_THRUST + vert - roll_in + pitch_in - yaw_dist
        fr = K_VERTICAL_THRUST + vert + roll_in + pitch_in + yaw_dist
        rl = K_VERTICAL_THRUST + vert - roll_in - pitch_in + yaw_dist
        rr = K_VERTICAL_THRUST + vert + roll_in - pitch_in - yaw_dist
        self.motors[0].setVelocity(fl)
        self.motors[1].setVelocity(-fr)
        self.motors[2].setVelocity(-rl)
        self.motors[3].setVelocity(rr)
