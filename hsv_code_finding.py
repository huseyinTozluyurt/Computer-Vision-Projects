import cv2
import numpy as np

def nothing(x):
    pass

cap = cv2.VideoCapture(0)


cv2.namedWindow('HSV Adjustments')
cv2.resizeWindow("HSV Adjustments", 1500, 1200)

cv2.createTrackbar('Lower - H', 'HSV Adjustments', 0, 180, nothing)  
cv2.createTrackbar('Lower - S', 'HSV Adjustments', 0, 180, nothing)
cv2.createTrackbar('Lower - V', 'HSV Adjustments', 0, 180, nothing)

cv2.createTrackbar('Upper - H', 'HSV Adjustments', 0, 180, nothing)
cv2.createTrackbar('Upper - S', 'HSV Adjustments', 0, 255, nothing)
cv2.createTrackbar('Upper - V', 'HSV Adjustments', 0, 255, nothing)

cv2.setTrackbarPos('Upper - H', 'HSV Adjustments', 180)
cv2.setTrackbarPos('Upper - S', 'HSV Adjustments', 255)
cv2.setTrackbarPos('Upper - V', 'HSV Adjustments', 255)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally for a mirror effect
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_h = cv2.getTrackbarPos('Lower - H', 'HSV Adjustments')
    lower_s = cv2.getTrackbarPos('Lower - S', 'HSV Adjustments')
    lower_v = cv2.getTrackbarPos('Lower - V', 'HSV Adjustments')

    upper_h = cv2.getTrackbarPos('Upper - H', 'HSV Adjustments')
    upper_s = cv2.getTrackbarPos('Upper - S', 'HSV Adjustments')
    upper_v = cv2.getTrackbarPos('Upper - V', 'HSV Adjustments')

    lower_bound = np.array([lower_h, lower_s, lower_v])
    upper_bound = np.array([upper_h, upper_s, upper_v])

    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # result = cv2.bitwise_and(frame, frame, mask=mask)

    # combined_display = np.hstack((frame, result))
    
    cv2.imshow('Webcam and Masked Result', mask)
    
    if cv2.waitKey(20) == ord('q'):
        break