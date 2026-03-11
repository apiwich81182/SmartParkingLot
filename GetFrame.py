import cv2

cap = cv2.VideoCapture('park3.mp4') 

success, img = cap.read()
if success:
    cv2.imwrite('parking_image3.jpg', img)
    print("Saved successfully")
cap.release()