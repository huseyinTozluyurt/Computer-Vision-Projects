import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1024,800))

    if not ret:
        print("Video not captured!")
        break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    adaptive_frame = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    cv2.imshow("Original Frame", frame)
    cv2.imshow("Adaptive Frame", adaptive_frame)






cap.release()
cv2.destroyAllWindows()