import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (1024,800))
    if not ret:
        print("Camera frame not found! ")
        break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    height, width, channels = frame.shape
    
    center_x = width // 2
    center_y = height // 2
    cv2.line(frame, (center_x, 0), (center_x, height), (0, 0, 255), 3)
    cv2.rectangle(frame, (center_x - 200, 100), (center_x + 200, height-100), (255,0,0), 3)
    cv2.line(frame, (0, center_y), (width, center_y), (0, 0, 255), 3)
    cv2.imshow("Frame", frame)


cap.release()
cv2.destroyAllWindows()