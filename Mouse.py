import cv2
import pickle

IMAGE_PATH = 'parking_image3.jpg' 
OUTPUT_FILE = 'ParkingPos3.pickle' 

# กำหนดขนาดของช่องจอดรถ
WIDTH = 40 # ความกว้าง
HEIGHT = 20 # ความสูง

try:
    with open(OUTPUT_FILE, 'rb') as f:
        posList = pickle.load(f)
except FileNotFoundError:
    posList = []

def mouseClick(event, x, y, flags, params):
    global posList
    
    # คลิกซ้าย: วางกรอบ
    if event == cv2.EVENT_LBUTTONDOWN:
        x_min, y_min = x, y
        x_max, y_max = x + WIDTH, y + HEIGHT
        
        posList.append((x_min, y_min, x_max, y_max))
        print(f"Added spot. Total: {len(posList)}")

    # คลิกขวา: ลบกรอบที่เมาส์ชี้อยู่
    elif event == cv2.EVENT_RBUTTONDOWN:
        for i, pos in enumerate(posList):
            x1, y1, x2, y2 = pos
            # เช็คว่าตำแหน่งเมาส์ที่คลิกขวา อยู่ในพื้นที่กรอบไหน
            if x1 < x < x2 and y1 < y < y2:
                posList.pop(i)
                print(f"Removed spot. Total: {len(posList)}")
                break

# ลูปแสดงผล
while True:
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("Error: Could not load image. เช็คชื่อไฟล์รูปภาพอีกครั้ง")
        break

    # วาดกรอบที่มีอยู่ทั้งหมด
    for pos in posList:
        cv2.rectangle(img, (pos[0], pos[1]), (pos[2], pos[3]), (255, 0, 255), 2)

    cv2.imshow("Image ROI Selector", img)
    cv2.setMouseCallback("Image ROI Selector", mouseClick)
    
    key = cv2.waitKey(1)
    if key & 0xFF == ord('s'): # กด s เพื่อเซฟ
        with open(OUTPUT_FILE, 'wb') as f:
            pickle.dump(posList, f)
        print(f"--- Saved {len(posList)} spots to {OUTPUT_FILE} ---")
    elif key & 0xFF == ord('q'): # กด q เพื่อออก
        break

cv2.destroyAllWindows()