import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Video not captured")
        break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blurred_frame, 50, 150)
    cv2.imshow("Edge Detection", edges)
    cv2.imshow("Original Frame", frame)

cap.release()
cv2.destroyAllWindows()

