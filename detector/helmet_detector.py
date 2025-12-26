from detector.motorcyclist_detector import MotorcyclistDetector
from models.helmet import get_helmet_model, get_helmet_infer_transform
from ultralytics import YOLO
import torch
import cv2
from utils.helper import iou, distance
from PIL import Image

class HelmetPresenceDetector(MotorcyclistDetector):
    """Detect motorcyclists and infer helmet presence using YOLO and a helmet classification model."""
    def __init__(self, yolo_model=YOLO("models/yolov8s.pt"), helmet_model=get_helmet_model(), helmet_transform=get_helmet_infer_transform(),device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(yolo_model)
        print("Loading Helmet(Resnet18) model...")
        self.helmet_model = helmet_model
        self.helmet_transform = helmet_transform
        self.device = device
    def infer_helmet(self, image, person_box):
        """Infer helmet presence for a given person bounding box in the image."""
        x1, y1, x2, y2, _ = person_box
        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            return None, 0.0

        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = Image.fromarray(crop)
        crop = self.helmet_transform(crop).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.helmet_model(crop)
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, 1)

        helmet = (pred.item() == 0)
        return helmet, conf.item()

    def associate(self, image):
        """Associate detected persons with motorcycles and infer helmet presence."""
        persons, motorcycles = self.get_person_motorcycle_boxes(image)
        results = []

        for mx1, my1, mx2, my2, mconf in motorcycles:
            m_box = (mx1, my1, mx2, my2)
            riders = []

            for p in persons:
                px1, py1, px2, py2, pconf = p
                p_box = (px1, py1, px2, py2)

                # Association rule
                if iou(p_box, m_box) > 0.05 or distance(p_box, m_box) < 80:
                    helmet, hconf = self.infer_helmet(image, p)

                    if helmet is None:
                        continue

                    riders.append({
                        "person_box": p_box,
                        "helmet": helmet,
                        "confidence": hconf
                    })

            if len(riders) > 0:
                results.append({
                    "motorcycle_box": m_box,
                    "riders": riders
                })

        return results

    def infer(self, frame):
        """Perform full inference: detect motorcyclists and infer helmet presence."""
        motorcyclists = self.associate(frame)
        # Draw results
        frame = self.draw_motorcyclists(frame, motorcyclists)
        return motorcyclists

    def draw_motorcyclists(
        self,
        frame,
        motorcyclists,
        conf_violate=0.70,
        conf_uncertain=0.60
    ):
        """Draw motorcyclist bounding boxes with helmet status."""
        for mc in motorcyclists:
            mx1, my1, mx2, my2 = mc["motorcycle_box"]

            # Start combined box with motorcycle
            cx1, cy1, cx2, cy2 = mx1, my1, mx2, my2

            has_violation = False
            has_uncertain = False

            for rider in mc["riders"]:
                px1, py1, px2, py2 = rider["person_box"]
                conf = rider["confidence"]
                helmet = rider["helmet"]

                # Expand combined box
                cx1 = min(cx1, px1)
                cy1 = min(cy1, py1)
                cx2 = max(cx2, px2)
                cy2 = max(cy2, py2)

                # Decision logic
                if conf < conf_uncertain:
                    has_uncertain = True
                elif not helmet and conf >= conf_violate:
                    has_violation = True

            # Final color decision
            if has_violation:
                color = (0, 0, 255)     # Red
                label = "Violation"
            elif has_uncertain:
                color = (0, 255, 255)   # Yellow
                label = "Uncertain"
            else:
                color = (0, 255, 0)     # Green
                label = "Helmet OK"

            # Draw combined bounding box
            cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), color, 3)

            # Label background
            cv2.rectangle(frame, (cx1, cy1 - 25), (cx1 + 140, cy1), color, -1)
            cv2.putText(
                frame,
                label,
                (cx1 + 5, cy1 - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return frame