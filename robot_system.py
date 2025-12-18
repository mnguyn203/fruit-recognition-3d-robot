import cv2
import numpy as np
import time
import threading
from queue import Queue
from typing import List, Optional
from robot_controller import RobotController, RobotCommand
from vision_processor import VisionProcessor, ObjectTracker, Object3D


class RobotVisionSystem:
    def __init__(self, debug=False):
        self.debug = debug

        # Configuration
        self.CLASS_MAPPING = {'quat': 1, 'quat_hong': 2}
        self.DROP_LOCATIONS_CAM = {
            'quat': np.array([0.202, -0.031, 0.5]),
            'quat_hong': np.array([0.205, -0.112, 0.5])
        }

        # Transformation matrix
        self.R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        self.t = np.array([0.268, -0.049, 0.58])

        # Components
        self.robot = RobotController(debug=debug)
        self.vision = VisionProcessor("C:/Users/artry/PycharmProjects/pythonProject12/test_robot/opp_quat_2/best_quat_v2.pt",
                                      debug=debug)
        self.tracker = ObjectTracker()

        # State
        self.detected_objects: List[Object3D] = []
        self.selected_index = -1
        self.picking = False
        self.auto_mode = False
        self.full_auto = False
        self.picked = set()
        self.running = True
        self.cmd_queue = Queue()

        # Convert drop locations
        self.drop_locations = {
            name: self.cam_to_base(loc) for name, loc in self.DROP_LOCATIONS_CAM.items()
        }

        if self.debug:
            print("🔢 Index Mapping:", self.CLASS_MAPPING)
            print(f"🔍 Shadow Filter: {'ON' if self.vision.shadow_filter_enabled else 'OFF'}")

    def cam_to_base(self, p: np.ndarray) -> np.ndarray:
        return self.R @ p + self.t

    def wait(self, seconds: float):
        for _ in range(int(seconds * 10)):
            if not self.running:
                break
            time.sleep(0.1)

    def pick_and_place(self, obj: Object3D):
        if self.picking:
            if self.debug: print("⚠️ Already picking")
            return

        self.picking = True
        try:
            if obj.name not in self.drop_locations:
                if self.debug: print(f"⚠️ No drop location for {obj.name}")
                self._simple_grab(obj)
                return

            pick_mm = self.cam_to_base(obj.center_3d) * 1000
            drop_mm = self.drop_locations[obj.name] * 1000

            steps = [
                (RobotCommand.OPEN, 1.0, "Open gripper"),
                (pick_mm, 3.0, "Move to pick"),
                (RobotCommand.CLOSE, 2.0, "Close gripper"),
                (pick_mm + [0, 0, 100], 2.0, "Lift"),
                (drop_mm + [0, 0, 100], 2.0, "Move to drop"),
                (drop_mm, 1.5, "Lower"),
                (RobotCommand.OPEN, 1.0, "Release"),
                (RobotCommand.HOME, 3.0, "Home")
            ]

            for step in steps:
                if not self.running:
                    break
                cmd, delay, desc = step
                if self.debug: print(f"🤖 {desc}")
                if isinstance(cmd, RobotCommand):
                    self.robot.send(cmd.value)
                else:
                    self.robot.move_to(cmd)
                self.wait(delay)

            self.picked.add(obj.name)
            print(f"✅ Completed: {obj.name}")

        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            self.picking = False

    def _simple_grab(self, obj: Object3D):
        pos_mm = self.cam_to_base(obj.center_3d) * 1000
        self.robot.send(RobotCommand.OPEN.value)
        self.wait(1.0)
        self.robot.move_to(pos_mm)
        self.wait(3.0)
        self.robot.send(RobotCommand.CLOSE.value)
        self.wait(2.0)

    def auto_worker(self):
        if self.debug: print("🚀 Full auto started")
        self.robot.send(RobotCommand.HOME.value)
        self.wait(3.0)

        while self.full_auto and self.running:
            next_obj = None
            for obj in self.detected_objects:
                if obj.name in self.drop_locations and obj.name not in self.picked:
                    next_obj = obj
                    break

            if not next_obj:
                print(f"🎉 Completed! Picked: {len(self.picked)}")
                self.full_auto = False
                break

            self.picked.add(next_obj.name)
            self.pick_and_place(next_obj)
            self.wait(3.0)

    def process_command(self, cmd: str):
        if self.picking and cmd not in ['Q', 'S', 'F', 'D']:
            if self.debug: print("⚠️ Busy picking")
            return

        handlers = {
            'Q': lambda: setattr(self, 'running', False),
            'H': lambda: self.robot.send(RobotCommand.HOME.value),
            'O': lambda: self.robot.send(RobotCommand.OPEN.value),
            'C': lambda: self.robot.send(RobotCommand.CLOSE.value),
            'A': self._toggle_auto,
            'N': self._start_full_auto,
            'S': self._stop_full_auto,
            'P': self._pick_selected,
            'G': self._grab_selected,
            'F': self._toggle_shadow_filter,
            'D': self._toggle_debug,
        }

        if cmd in handlers:
            handlers[cmd]()
        elif cmd.isdigit():
            self._select_object(int(cmd))
        else:
            self._manual_move(cmd)

    def _toggle_debug(self):
        self.debug = not self.debug
        self.robot.debug = self.debug
        self.vision.debug = self.debug
        print(f"🐛 Debug Mode: {'ON' if self.debug else 'OFF'}")

    def _toggle_shadow_filter(self):
        self.vision.shadow_filter_enabled = not self.vision.shadow_filter_enabled
        print(f"🔍 Shadow Filter: {'ON' if self.vision.shadow_filter_enabled else 'OFF'}")

    def _toggle_auto(self):
        self.auto_mode = not self.auto_mode
        print(f"🔄 Auto: {'ON' if self.auto_mode else 'OFF'}")

    def _start_full_auto(self):
        if not self.full_auto:
            self.full_auto = True
            self.picked.clear()
            threading.Thread(target=self.auto_worker, daemon=True).start()

    def _stop_full_auto(self):
        self.full_auto = False
        print("⏹️ Full auto stopped")

    def _pick_selected(self):
        if self.selected_index != -1:
            obj = self._find_object(self.selected_index)
            if obj:
                threading.Thread(target=self.pick_and_place, args=(obj,)).start()

    def _grab_selected(self):
        if self.selected_index != -1:
            obj = self._find_object(self.selected_index)
            if obj:
                self._simple_grab(obj)

    def _select_object(self, index: int):
        obj = self._find_object(index)
        if obj:
            self.selected_index = index
            print(f"🎯 Selected [{index}]: {obj.name} ({obj.confidence * 100:.0f}%)")
        else:
            print(f"⚠️ Object [{index}] not detected")

    def _find_object(self, index: int) -> Optional[Object3D]:
        for obj in self.detected_objects:
            if obj.fixed_index == index:
                return obj
        return None

    def _manual_move(self, cmd: str):
        try:
            coords = cmd.replace(',', ' ').split()
            if len(coords) == 3:
                p = np.array([float(x) for x in coords])
                self.robot.move_to(self.cam_to_base(p) * 1000)
        except:
            pass

    def draw_ui(self, img: np.ndarray):
        h, w = img.shape[:2]

        # Draw detected objects
        for obj in self.detected_objects:
            u, v = self.vision.project_to_image(obj.center_3d)
            selected = obj.fixed_index == self.selected_index
            picked = obj.name in self.picked

            # Color coding
            if picked:
                color = (128, 128, 128)
            elif obj.name in self.drop_locations:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)

            if selected:
                color = (0, 0, 255)

            # Draw contours
            if obj.mask is not None:
                contours, _ = cv2.findContours(obj.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                thickness = 3 if selected else 2
                cv2.drawContours(img, contours, -1, color, thickness)

            # Draw center
            center_color = (0, 255, 255) if selected else (255, 255, 0)
            cv2.circle(img, (u, v), 8 if selected else 6, center_color, -1)

            # Label with confidence percentage
            label = f"[{obj.fixed_index}]{obj.name} {obj.confidence * 100:.0f}%"
            if picked:
                label += " ✓"

            cv2.putText(img, label, (u - 30, v - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Status panel
        lighting = self.vision.get_lighting_condition()
        status = [
            f"Objects: {len(self.detected_objects)}",
            f"Lighting: {lighting}",
            f"Filter: {'ON' if self.vision.shadow_filter_enabled else 'OFF'}",
            f"Debug: {'ON' if self.debug else 'OFF'}",
            f"Auto: {'ON' if self.auto_mode else 'OFF'}",
            f"Full Auto: {'ON' if self.full_auto else 'OFF'}",
            f"Picked: {len(self.picked)}"
        ]

        for i, text in enumerate(status):
            y_pos = 30 + i * 25
            text_color = (0, 255, 255)
            if i == 2:  # Filter status
                text_color = (0, 255, 0) if self.vision.shadow_filter_enabled else (0, 0, 255)
            elif i == 3:  # Debug status
                text_color = (0, 255, 0) if self.debug else (0, 0, 255)
            cv2.putText(img, text, (w - 200, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    def run(self):
        print("\n🎮 CONTROLS:")
        print("1,2: Select | P: Pick&Place | G: Grab | F: Filter | D: Debug | A: Auto | N: Full Auto | Q: Quit\n")

        threading.Thread(target=self._input_thread, daemon=True).start()

        try:
            while self.running:
                while not self.cmd_queue.empty():
                    self.process_command(self.cmd_queue.get())

                color, depth = self.vision.get_frames()
                if color is None:
                    continue

                detections = self.vision.detect_objects(color, depth)
                self.detected_objects = self.tracker.update(detections, self.CLASS_MAPPING)

                self.draw_ui(color)
                cv2.imshow("Robot Vision System (Clean)", color)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False

        finally:
            self.cleanup()

    def _input_thread(self):
        while self.running:
            try:
                cmd = input().strip().upper()
                if cmd:
                    self.cmd_queue.put(cmd)
            except:
                break

    def cleanup(self):
        self.running = False
        self.full_auto = False
        cv2.destroyAllWindows()
        self.vision.cleanup()
        self.robot.close()
        if self.debug: print("✅ Cleanup complete")