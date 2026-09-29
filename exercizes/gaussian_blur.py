import cv2
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (800,500))
    if not ret:
        print("Frame not captured !")
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (9,9), 0)
    cv2.imshow("Original Frame", frame)
    cv2.imshow("Gray Frame", gray_frame)
    cv2.imshow("Blurred Frame", blurred_frame)

cap.release()
cv2.destroyAllWindows()
    


