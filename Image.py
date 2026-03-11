import cv2
import numpy as np
import pickle
import os

# 1. ตั้งค่าพารามิเตอร์เฉพาะของ Cam 3
p = {
    'm_blur': 3, 'g_blur': 5, 'c_low': 30, 'c_high': 170, 'd_iter': 2, 'limit': 120, 
    'pickle': 'ParkingPos3.pickle', 'video': 'park3.mp4', 'ref_img': 'parking_image3.jpg'
}

# 2. เตรียมระบบ Feature Matching (ORB)
ref_img = cv2.imread(p['ref_img'], cv2.IMREAD_GRAYSCALE)
if ref_img is None:
    print(f"Error: Could not load reference image '{p['ref_img']}' ")
    exit()

orb = cv2.ORB_create(1000)
kp_ref, des_ref = orb.detectAndCompute(ref_img, None)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# 3. โหลดพิกัดช่องจอด
if not os.path.exists(p['pickle']) or not os.path.exists(p['video']):
    print("Error: Pickle file or Video file not found!")
    exit()
    
with open(p['pickle'], 'rb') as f:
    posList = pickle.load(f)

# 4. เปิดวิดีโอและเตรียม VideoWriter
cap = cv2.VideoCapture(p['video'])
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps == 0: fps = 20

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('smart_parking_cam3_dynamic.mp4', fourcc, fps, (width, height))

print("Processing Cam3 with Dynamic ROI... PRESS 'q' TO STOP")

while True:
    success, img = cap.read()
    if not success:
        print("Video ended.")
        break
        
    img_display = img.copy()
    
    # ==========================================
    # ส่วนที่ 1: หาเมทริกซ์การเคลื่อนที่ (Homography)
    # ==========================================
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp_curr, des_curr = orb.detectAndCompute(gray, None)
    
    H = None
    if des_curr is not None and len(des_curr) > 0:
        matches = matcher.match(des_ref, des_curr)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:50]
        
        if len(good_matches) >= 4:
            src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_curr[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # ==========================================
    # ส่วนที่ 2: Image Processing Pipeline (DIP)
    # ==========================================
    median = cv2.medianBlur(gray, p['m_blur'])
    blur = cv2.GaussianBlur(median, (p['g_blur'], p['g_blur']), 1)
    canny = cv2.Canny(blur, p['c_low'], p['c_high'])
    dilate = cv2.dilate(canny, np.ones((3, 3), np.uint8), iterations=p['d_iter'])
    
    # ==========================================
    # ส่วนที่ 3: ตรวจสอบพิกเซลในกรอบที่เอียงแล้ว
    # ==========================================
    spaceCounter = 0
    for pos in posList:
        px1, py1, px2, py2 = map(int, pos)
        
        # แปลงพิกัด 4 มุมตามการสั่นของกล้อง
        pts = np.float32([[px1, py1], [px2, py1], [px2, py2], [px1, py2]]).reshape(-1, 1, 2)
        if H is not None:
            poly_pts = np.int32(cv2.perspectiveTransform(pts, H))
        else:
            poly_pts = np.int32(pts)
            
        # สร้าง Bounding Box ครอบรูปหลายเหลี่ยมเพื่อตีกรอบพื้นที่ทำงาน (ลดการกินสเปคเครื่อง)
        bx, by, bw, bh = cv2.boundingRect(poly_pts)
        bx, by = max(0, bx), max(0, by)
        bw = min(dilate.shape[1] - bx, bw)
        bh = min(dilate.shape[0] - by, bh)
        
        if bw <= 0 or bh <= 0: continue
            
        # ตัดภาพเฉพาะส่วนขอบสีขาว (Dilate)
        dilate_crop = dilate[by:by+bh, bx:bx+bw]
        
        # สร้าง Mask สีดำ และระบายสีขาวเฉพาะในรูปหลายเหลี่ยม
        mask = np.zeros((bh, bw), dtype=np.uint8)
        local_poly = poly_pts - [bx, by] # ขยับพิกัดมาอยู่ในจุดอ้างอิงของการ Crop
        cv2.fillPoly(mask, [local_poly], 255)
        
        # หด Mask ลงนิดนึง (Erosion) เพื่อทำหน้าที่แทน Pad 15% ในโค้ดเก่า (กันขอบเส้นจอดรถ)
        mask = cv2.erode(mask, np.ones((5,5), np.uint8), iterations=1)
        
        # นำ Mask ทาบทับภาพ Dilate และนับจุดสีขาว
        masked_crop = cv2.bitwise_and(dilate_crop, mask)
        count = cv2.countNonZero(masked_crop)
        
        # วาดกราฟิกแสดงผล
        color, thick = ((0, 255, 0), 2) if count < p['limit'] else ((0, 0, 255), 1)
        if count < p['limit']: spaceCounter += 1
        
        cv2.polylines(img_display, [poly_pts], True, color, thick)
        
        # หาจุดกึ่งกลางของ Polygon เพื่อวาดตัวเลข
        cx = int(np.mean(poly_pts[:, 0, 0]))
        cy = int(np.mean(poly_pts[:, 0, 1]))
        cv2.putText(img_display, str(count), (cx - 15, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

    # แสดงผลรวม
    cv2.rectangle(img_display, (0, 0), (250, 60), (0, 0, 0), -1)
    cv2.putText(img_display, f"Free: {spaceCounter}/{len(posList)}", (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Cam3 Dynamic DIP", img_display)
    out.write(img_display)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# คืนทรัพยากรระบบ
cap.release()
out.release()
cv2.destroyAllWindows()
print("Success! Saved to 'smart_parking_cam3_dynamic.mp4'")