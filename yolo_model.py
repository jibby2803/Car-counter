from ultralytics import YOLO
import cv2

model = YOLO('./YOLO-weight/yolov8n.pt')
result = model('./assets/images/img2.jpg',show=True)
cv2.waitKey(0)
