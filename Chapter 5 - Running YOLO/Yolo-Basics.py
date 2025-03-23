from ultralytics import YOLO
import cv2

model = YOLO("../Yolo-Weights/yolov8n.pt")
result = model("Images/1.png" , show =True)
#print("THe result is " , result)
# for r in result :
#     boxes = r.boxes
#     print("-----------------------------------------------------------------------------------")
#     print("The boxes are " , boxes)
#     print("-----------------------------------------------------------------------------------")

cv2.waitKey(0)