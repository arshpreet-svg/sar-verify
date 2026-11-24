import json


class Radio:
    def __init__(self, robot, me, step):
        self.me = me
        self.emitter = robot.getDevice("emitter")
        self.receiver = robot.getDevice("receiver")
        self.receiver.enable(step)
        self.inbox = []

    def send(self, to, mtype, **payload):
        msg = {"to": to, "from": self.me, "type": mtype}
        msg.update(payload)
        self.emitter.send(json.dumps(msg).encode("utf-8"))

    def poll(self):
        out = []
        while self.receiver.getQueueLength() > 0:
            raw = self.receiver.getString()
            self.receiver.nextPacket()
            try:
                msg = json.loads(raw)
            except ValueError:
                continue
            if msg.get("to") in (self.me, "all"):
                out.append(msg)
        return out
