from ultralytics import YOLO
import cv2
from models.helmet import get_helmet_model, get_helmet_infer_transform
import argparse
from detector.helmet_detector import HelmetPresenceDetector

# ------------------
# Main
# ------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, help="Path to image")
    args = parser.parse_args()

    if args.img:
        frame = cv2.imread(args.img)
        helmet_presence_detector = HelmetPresenceDetector(YOLO("models/yolov8s.pt"),
                                                         get_helmet_model("weights/helmet_resnet18_best.pth"),
                                                         get_helmet_infer_transform(224))
        motorcyclists = helmet_presence_detector.infer(frame)
        cv2.imwrite("helmet_detection_output.jpg", frame)
        cv2.imshow("Helmet Detection", frame)
        cv2.waitKey(0)
    else:
        print("Please specify --img")