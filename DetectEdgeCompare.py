import cv2
import numpy as np

# ฟังก์ชันสำหรับใส่ป้ายชื่อบนรูป
def add_label(img, text):
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) # แปลงเป็น 3 Channel เพื่อให้ใส่สีได้
    cv2.rectangle(img_bgr, (0, 0), (350, 40), (0, 0, 0), cv2.FILLED)
    cv2.putText(img_bgr, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return img_bgr

# เปิดวิดีโอลานจอดรถ
cap = cv2.VideoCapture("park1.mp4")

print("qqq")

while True:
    success, img = cap.read()
    if not success:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # 1. ภาพต้นฉบับ (ทำเป็นขาวดำและเบลอเล็กน้อยเพื่อความยุติธรรมให้ทุกฟิลเตอร์)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)

    # 2. Canny Edge (พระเอกของเรา)
    canny = cv2.Canny(blur, 70, 180)

    # 3. Sobel (รวมแกน X และ Y)
    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.convertScaleAbs(cv2.bitwise_or(sobelx, sobely))

    # 4. Laplacian (อนุพันธ์อันดับ 2)
    laplacian = cv2.convertScaleAbs(cv2.Laplacian(blur, cv2.CV_64F))

    # 5. Adaptive Threshold (Gaussian)
    # ใช้ Block size 15, ตัดค่าคงที่ 2 (Invert สีให้ขอบ/วัตถุเป็นสีขาวเหมือน Canny)
    adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 15, 2)

    # 6. Otsu's Threshold (Global Thresholding อัตโนมัติ)
    # สลับสีให้วัตถุเป็นสีขาว (THRESH_BINARY_INV)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # --- เตรียมภาพสำหรับแสดงผล (ย่อขนาดลง 40% จะได้ไม่ล้นจอ) ---
    scale = 0.4
    h, w = gray.shape
    new_w, new_h = int(w * scale), int(h * scale)

    # ใส่ป้ายชื่อและย่อขนาด
    img1 = cv2.resize(add_label(gray, "1. Original (Gray+Blur)"), (new_w, new_h))
    img2 = cv2.resize(add_label(canny, "2. Canny (Clean & Thin)"), (new_w, new_h))
    img3 = cv2.resize(add_label(sobel_combined, "3. Sobel (Thick & Messy)"), (new_w, new_h))
    img4 = cv2.resize(add_label(laplacian, "4. Laplacian (Extremely Noisy)"), (new_w, new_h))
    img5 = cv2.resize(add_label(adaptive, "5. Adaptive (Catches Road Texture)"), (new_w, new_h))
    img6 = cv2.resize(add_label(otsu, "6. Otsu (Fails on Uneven Light)"), (new_w, new_h))

    # นำมาต่อกันเป็นตาราง 2 แถว 3 คอลัมน์
    row1 = np.hstack([img1, img2, img3])
    row2 = np.hstack([img4, img5, img6])
    grid = np.vstack([row1, row2])

    cv2.imshow("Edge Detection & Thresholding Comparison", grid)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()