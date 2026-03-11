import cv2
import pickle
from ultralytics import YOLO

# 1. โหลดโมเดล
model = YOLO('runs/detect/train7/weights/best.pt')

# 2. โหลดพิกัดช่องจอด
with open('ParkingPos3.pickle', 'rb') as f:
    posList = pickle.load(f)

# 3. เปิดวิดีโอลานจอดรถ
cap = cv2.VideoCapture('park3.mp4')

# ดึงขนาดความกว้าง ความสูง และ FPS ของต้นฉบับมาใช้
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps == 0: fps = 30

# ตั้งค่า Codec (mp4v)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('smart_parking_output2.mp4', fourcc, fps, (frame_width, frame_height))

print("Starting... PRESS 'q' TO STOP AND SAVE VIDEO")

while True:
    success, img = cap.read()
    
    # ถ้าวิดีโอจบ ให้หยุดการทำงานเพื่อเซฟไฟล์
    if not success:
        print("End of video. Saving file...")
        break

    # 4. ให้ AI ตรวจจับรถ
    results = model.predict(img, classes=[3, 4, 5, 8], conf=0.15, verbose=False)
    car_boxes = results[0].boxes.xyxy.cpu().numpy()

    spaceCounter = 0

    # 5. เช็คว่ามีรถอยู่ในช่องจอดไหม
    for pos in posList:
        px1, py1, px2, py2 = map(int, pos)
        is_occupied = False
        
        for car in car_boxes:
            cx1, cy1, cx2, cy2 = map(int, car)
            car_center_x = int((cx1 + cx2) / 2)
            car_center_y = int((cy1 + cy2) / 2)
            
            if (px1 < car_center_x < px2) and (py1 < car_center_y < py2):
                is_occupied = True
                break
                
        color = (0, 0, 255) if is_occupied else (0, 255, 0)
        thickness = 2
        if not is_occupied: spaceCounter += 1
            
        cv2.rectangle(img, (px1, py1), (px2, py2), color, thickness)
        
    cv2.rectangle(img, (0, 0), (250, 60), (0, 0, 0), -1)
    cv2.putText(img, f"Free: {spaceCounter}/{len(posList)}", (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Custom VisDrone Parking Detection", img)
    
    out.write(img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดการทำงานและรันคำสั่งเซฟไฟล์
cap.release()
out.release() 
cv2.destroyAllWindows()

print("Video saved successfully")