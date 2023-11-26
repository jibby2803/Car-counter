from ultralytics import YOLO
import cv2

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

cap = cv2.VideoCapture('./assets/videos/cars.mp4')
if cap.isOpened():
    width = cap.get(3) 
    height = cap.get(4)
    print(f'width = {width}, height = {height}')
    
mask = cv2.imread('./assets/images/mask.jpg')
mask = cv2.resize(mask, (int(width), int(height)))

while True:
    flag, img = cap.read()
    if not flag:
        break
    mask_region = cv2.bitwise_and(mask, img)
    
    results = model(mask_region, stream=True)
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = int(box.cls[0])
            cv2.putText(img, classes[label], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            
            
    cv2.imshow('image', img)
    cv2.waitKey(1)
   