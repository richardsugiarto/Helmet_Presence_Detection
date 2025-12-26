from ultralytics import YOLO

class MotorcyclistDetector:
    """Detect persons and motorcycles in images using a YOLO model."""
    def __init__(self, yolo_model=YOLO("models/yolov8s.pt")):
        print("Loading YOLO model...")
        self.yolo_model = yolo_model

    def get_person_motorcycle_boxes(self, img):
        """Detect persons and motorcycles in the image and return their bounding boxes."""
        persons = []
        motorcycles = []
        results = self.yolo_model(img, conf=0.4)

        for box in results[0].boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls == 0:  # person
                persons.append((x1, y1, x2, y2, conf))

            elif cls == 3:  # motorcycle
                motorcycles.append((x1, y1, x2, y2, conf))

        return persons, motorcycles
