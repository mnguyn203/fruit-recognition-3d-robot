import serial
import time
from enum import Enum


class RobotCommand(Enum):
    HOME = "H"
    OPEN = "O"
    CLOSE = "C"
    MOVE = "G"


class RobotController:
    def __init__(self, port="COM4", baudrate=115200, debug=False):
        self.debug = debug
        self.ser = None
        try:
            self.ser = serial.Serial(port, baudrate, timeout=1)
            time.sleep(2)
            if self.debug: print("✅ Serial connected")
        except Exception as e:
            if self.debug: print(f"⚠️ Serial error: {e}")

    def send(self, cmd: str, delay: float = 0.1):
        if not self.ser:
            if self.debug: print(f"⚠️ No serial: {cmd}")
            return
        try:
            self.ser.write((cmd + "\n").encode("utf-8"))
            if self.debug: print(f"→ Robot: {cmd}")
            time.sleep(delay)
        except Exception as e:
            if self.debug: print(f"⚠️ Send error: {e}")

    def move_to(self, pos_mm):
        self.send(f"G{pos_mm[0]:.1f},{pos_mm[1]:.1f},{pos_mm[2]:.1f}")

    def close(self):
        if self.ser:
            self.ser.close()