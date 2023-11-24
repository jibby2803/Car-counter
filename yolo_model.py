from ultralytics import YOLO
import cv2

# model = YOLO('./YOLO-weight/yolov8n.pt')

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
while True:
    flag, img = cap.read()
    cv2.imshow('image', img)
    cv2.waitKey(1)
    