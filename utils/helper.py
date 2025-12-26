import math

def center_of(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def distance(boxA, boxB): # euclidian distance between centers
    cxA, cyA = center_of(boxA)
    cxB, cyB = center_of(boxB)
    return math.sqrt((cxA-cxB)**2 + (cyA-cyB)**2)

def iou(boxA, boxB): # area of overlap / area of union
    # standard IoU formula
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB-xA) * max(0, yB-yA)
    boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return interArea / (boxAArea + boxBArea - interArea + 1e-6)