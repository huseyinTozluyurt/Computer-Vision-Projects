import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1024, 800))

    if not ret:
        print("Camera error.")
        break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_scale, 80, 255, cv2.THRESH_BINARY)
    cv2.imshow("Original Frame", frame)
    cv2.imshow("Thresholded Frame", thresh)

cap.release()
cv2.destroyAllWindows()
        

