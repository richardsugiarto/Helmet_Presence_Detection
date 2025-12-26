import cv2
from detector.motorcyclist_detector import MotorcyclistDetector
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, help="Path to image")
    args = parser.parse_args()

    if args.img:
        img = cv2.imread(args.img)

        motorcyclists_detector = MotorcyclistDetector()

        persons, motorcycles = motorcyclists_detector.get_person_motorcycle_boxes(img)

        for mx1, my1, mx2, my2, mconf in motorcycles:
            cv2.rectangle(img, (mx1, my1), (mx2, my2), (0,255,0), 2)
            cv2.putText(img, f"Motorcycle {mconf:.2f}", (mx1, my1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        for px1, py1, px2, py2, pconf in persons:
            cv2.rectangle(img, (px1, py1), (px2, py2), (0,0,255), 2)
            cv2.putText(img, f"Person {pconf:.2f}", (px1, py1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

        cv2.imshow("Person Motorcycle Inference", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Please specify --img")