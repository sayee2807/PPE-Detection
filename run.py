from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture("../Videos/3.mp4")
model = YOLO("../model/best.pt")

classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
              'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']

PPE_CLASSES = ['Hardhat','Mask','NO-Hardhat','NO-Mask','NO-Safety Vest','Safety Vest']

while True:
    success, img = cap.read()
    if not success:
        break

    results = model(img, stream=True)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass not in PPE_CLASSES:
                continue

            conf = round(float(box.conf[0]), 2)
            if conf < 0.5:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if currentClass in ['NO-Hardhat','NO-Safety Vest','NO-Mask']:
                color = (0,0,255) #RED
            else:
                color = (0,255,0) #GREEN

            cvzone.putTextRect(img, f'{currentClass} {conf}',
                               (max(0,x1), max(35,y1)), scale=1, thickness=1,
                               colorB=color, colorT=(255,255,255), colorR=color, offset=5)

            cv2.rectangle(img,(x1,y1),(x2,y2),color,3)

    cv2.imshow("PPE Detection Only", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()


