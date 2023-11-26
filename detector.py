from ultralytics import YOLO
import cv2
import numpy as np
from sort import *

model = YOLO('yolov8n.pt')

classes = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
            "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
            "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
            "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
            "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
            "teddy bear", "hair drier", "toothbrush"]

# READ YOUR VIDEO HERE
# -----------------------------------------------
cap = cv2.VideoCapture('./assets/videos/cars.mp4') # change path to your video file path
# -----------------------------------------------

if cap.isOpened():
    width = cap.get(3) 
    height = cap.get(4)
    # print(f'width = {width}, height = {height}')
# output = cv2.VideoWriter('./result/result.mp4',
#                          cv2.VideoWriter_fourcc(*'mp4v'),
#                          30, (int(width), int(height)))

# CHANGE YOUR MASK HERE 
# -----------------------------------------------
mask = cv2.imread('./assets/images/mask.jpg') # change path to your mask file path
# -----------------------------------------------

mask = cv2.resize(mask, (int(width), int(height)))

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.4)

# CHANGE THE COORDINATES OF THE COUNTING LINE 
# ------------------------------------------------
limit = [380, 290, 700, 290] # adjust the [x1, y1, x2, y2] according to your video and mask where (x1, y1), (x2, y2) are the coordinates starting and ending point of counting line
# ------------------------------------------------

total_count = []
count = 0

while True:
    flag, img = cap.read()
    if not flag:
        break
    mask_region = cv2.bitwise_and(mask, img)
    
    results = model(mask_region, stream=True)
    
    detections = np.empty((0, 5))
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = int(box.cls[0])
            # cv2.putText(img, classes[label], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            conf = box.conf[0]
            if conf > 0.4 and (classes[label] == 'bicycle' or classes[label]=='car' or classes[label]=='motorbike' \
                or classes[label]=='bus' or classes[label]=='truck'):
                # cv2.putText(img, f'{classes[label]}{conf:.2f}', (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255), 1)           
                current_stat = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, current_stat))
    
    result_tracker = tracker.update(detections)
    cv2.line(img, (limit[0], limit[1]), (limit[2], limit[3]), (0, 0, 255), 4)
    
    for trk in result_tracker:
        x1, y1, x2, y2, id = map(int, trk)
        cx = (x1+x2)//2
        cy = (y1+y2)//2
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(img, (cx, cy), 4, (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'id: {id}', (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 1)
        if (limit[0]<= cx <= limit[2]) and (limit[1]-30 <= cy <= limit[3]+30):
            if id not in total_count:
                total_count.append(id)
                count += 1
                cv2.line(img, (limit[0], limit[1]), (limit[2], limit[3]), (0, 255, 0), 4)
    
    cv2.putText(img, f'total count: {count}', (40, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)           
    
    # output.write(img)
    cv2.imshow('result', img)
    cv2.waitKey(1)
    
cap.release() 
# output.release()
cv2.destroyAllWindows() 
    

   