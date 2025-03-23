from ultralytics import YOLO
import cv2
import cvzone
import math

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# For WEBCAM
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap.set(3, 640)
# cap.set(4,480)

cap = cv2.VideoCapture("../Videos/bikes.mp4")

model = YOLO("../Yolo-Weights/yolov10n.pt")

while True:
    success , img = cap.read()
    result = model(img , stream = True)
    # print(result)
    for r in result :
        boxes = r.boxes
        # print(boxes)
        for box in boxes:
            # Bounding Box
            x1 , y1 , x2 , y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1) , int(y1) , int(x2) , int(y2)
            print(x1 , y1 , x2 , y2)
            # cv2.rectangle(img , (x1 , y1) , (x2 , y2) , (0 , 255 , 0 ) , 3)
            w , h = x2 - x1 , y2 - y1
            cvzone.cornerRect(img , (x1 , y1 , w, h))

            #COnfidence
            conf = math.floor((box.conf[0]*100))/100

            #Class Name
            cls = int(box.cls[0])

            cvzone.putTextRect(img , f'{classNames[cls]} {conf}' , (max(0 , x1) , max(30 , y1)) , scale = 0.7 , thickness=1)


    cv2.imshow("Image" , img)
    cv2.waitKey(1)


