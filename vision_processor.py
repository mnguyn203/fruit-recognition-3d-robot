import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import torch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Object3D:
    name: str
    fixed_index: int
    center_3d: np.ndarray
    size_3d: List[float]
    confidence: float
    distance: float
    mask: Optional[np.ndarray] = None


class ObjectTracker:
    def __init__(self, max_distance=0.12, max_missed=6, smoothing=0.6):
        self.tracks = {}
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.smoothing = smoothing

    def update(self, detections: List[Dict], class_mapping: Dict[str, int]) -> List[Object3D]:
        detected_classes = set()

        for det in detections:
            name = det['name']
            detected_classes.add(name)

            if name in self.tracks:
                track = self.tracks[name]
                alpha = self.smoothing
                track['center_3d'] = alpha * det['center_3d'] + (1 - alpha) * track['center_3d']
                track['size_3d'] = det['size_3d']
                track['confidence'] = det['confidence']
                track['missed'] = 0
                track['age'] += 1
                if 'mask' in det:
                    track['mask'] = det['mask']
            else:
                self.tracks[name] = {
                    'fixed_index': class_mapping.get(name, 0),
                    'name': name,
                    'center_3d': np.array(det['center_3d'], dtype=float),
                    'size_3d': det['size_3d'],
                    'confidence': det['confidence'],
                    'missed': 0,
                    'age': 1,
                    'mask': det.get('mask')
                }

        # Update missed counts
        for name in list(self.tracks.keys()):
            if name not in detected_classes:
                self.tracks[name]['missed'] += 1
                if self.tracks[name]['missed'] > self.max_missed:
                    del self.tracks[name]

        # Convert to Object3D list
        objects = []
        for track in sorted(self.tracks.values(), key=lambda x: x['fixed_index']):
            objects.append(Object3D(
                name=track['name'],
                fixed_index=track['fixed_index'],
                center_3d=track['center_3d'],
                size_3d=track['size_3d'],
                confidence=track['confidence'],
                distance=track['center_3d'][2],
                mask=track.get('mask')
            ))
        return objects


class VisionProcessor:
    def __init__(self, model_path: str, debug=False):
        self.debug = debug
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)
        self.model.to(self.device)
        if self.debug: print(f"✅ YOLO on {self.device}")

        # Setup camera
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)

        # Get intrinsics
        intr = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.fx, self.fy = intr.fx, intr.fy
        self.cx, self.cy = intr.ppx, intr.ppy
        if self.debug: print(f"✅ Camera: fx={self.fx:.2f}, fy={self.fy:.2f}")

        # Shadow filtering parameters
        self.shadow_filter_enabled = True
        self.min_depth_points = 50
        self.max_depth_std = 0.05
        self.min_brightness = 0.25
        self.confidence_boost = 0.1
        self.brightness_history = []

    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None, None

        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data()).astype(np.float32)
        return color, depth

    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray) / 255.0
        self.brightness_history.append(brightness)
        if len(self.brightness_history) > 10:
            self.brightness_history.pop(0)

        # Apply CLAHE
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Gamma correction for dark images
        if brightness < 0.3:
            gamma = 0.6
            table = np.array([((i / 255.0) ** (1 / gamma)) * 255 for i in range(256)]).astype("uint8")
            enhanced = cv2.LUT(enhanced, table)

        return enhanced

    def is_shadow(self, mask: np.ndarray, depth: np.ndarray, image: np.ndarray) -> Tuple[bool, float]:
        shadow_score = 0.0

        # Depth check
        points_3d = self._get_3d_points(mask, depth)
        if len(points_3d) < 30:
            shadow_score += 0.6
        else:
            z_std = np.std(points_3d[:, 2]) if len(points_3d) > 1 else 0
            if z_std > 0.08:
                shadow_score += 0.5

        # Brightness check
        masked_region = cv2.bitwise_and(image, image, mask=mask)
        gray_masked = cv2.cvtColor(masked_region, cv2.COLOR_BGR2GRAY)
        valid_pixels = gray_masked[mask > 0]

        if len(valid_pixels) > 0:
            brightness = np.mean(valid_pixels) / 255.0
            if brightness < 0.15:
                shadow_score += 0.4
        else:
            shadow_score += 0.3

        return shadow_score > 0.7, shadow_score

    def get_lighting_condition(self) -> str:
        if not self.brightness_history:
            return "unknown"
        avg_brightness = np.mean(self.brightness_history)
        if avg_brightness < 0.2:
            return "very_dark"
        elif avg_brightness < 0.4:
            return "dark"
        elif avg_brightness < 0.6:
            return "normal"
        else:
            return "bright"

    def detect_objects(self, color: np.ndarray, depth: np.ndarray) -> List[Dict]:
        enhanced_color = self.enhance_image(color) if self.shadow_filter_enabled else color
        results = self.model(enhanced_color, device=self.device, verbose=False, imgsz=640)
        detections = []

        if not results or results[0].masks is None:
            if self.debug and results and results[0].boxes is not None:
                print(f"📦 Found {len(results[0].boxes)} box detections (no masks)")
            return detections

        total_detections = len(results[0].masks.data)
        filtered_count = 0
        no_depth_count = 0

        for i, mask in enumerate(results[0].masks.data):
            cls_id = int(results[0].boxes.cls[i].cpu().numpy())
            conf = float(results[0].boxes.conf[i].cpu().numpy())
            name = results[0].names[cls_id]

            # Process mask
            mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else mask
            mask_np = cv2.resize((mask_np * 255).astype(np.uint8), (color.shape[1], color.shape[0]))

            # Calculate 3D points
            points_3d = self._get_3d_points(mask_np, depth)
            if len(points_3d) == 0:
                no_depth_count += 1
                continue

            # Shadow filtering
            if self.shadow_filter_enabled:
                is_shadow_result, shadow_score = self.is_shadow(mask_np, depth, enhanced_color)
                if is_shadow_result and shadow_score > 0.7:
                    filtered_count += 1
                    continue

            # Boost confidence for good depth
            z_std = np.std(points_3d[:, 2]) if len(points_3d) > 1 else 0
            if z_std < 0.02 and len(points_3d) > 100:
                conf = min(1.0, conf + self.confidence_boost)

            # Calculate 3D properties
            mins = points_3d.min(axis=0)
            maxs = points_3d.max(axis=0)
            center_3d = (mins + maxs) / 2
            # center_3d[2] = np.floor(center_3d[2] * 100) / 100


            detections.append({
                'name': name,
                'confidence': conf,
                'center_3d': center_3d,
                'size_3d': list(maxs - mins),
                'mask': mask_np
            })

        # Only show summary if debug mode or if filtering occurred
        if self.debug or filtered_count > 0 or no_depth_count > 0:
            print(f"📊 Objects: {len(detections)}/{total_detections}")
            if filtered_count > 0:
                print(f"  - {filtered_count} shadow filtered")
            if no_depth_count > 0:
                print(f"  - {no_depth_count} no depth data")

        return detections

    def _get_3d_points(self, mask: np.ndarray, depth: np.ndarray) -> np.ndarray:
        points = []
        v_indices, u_indices = np.where(mask > 0)

        for v, u in zip(v_indices, u_indices):
            z = depth[v, u] / 1000.0
            if 0.1 < z <= 3.0:
                x = (u - self.cx) * z / self.fx
                y = (v - self.cy) * z / self.fy
                points.append([x, y, z])

        return np.array(points)

    def project_to_image(self, point_3d: np.ndarray) -> Tuple[int, int]:
        z = max(point_3d[2], 0.0001)
        u = int(self.cx + point_3d[0] * self.fx / z)
        v = int(self.cy + point_3d[1] * self.fy / z)
        return u, v

    def cleanup(self):
        self.pipeline.stop()