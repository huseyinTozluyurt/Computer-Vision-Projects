import cv2
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (1024, 800))
    height, width, channels = frame.shape
    frame = frame[height//2:height, 0:width]
    if not ret:
        print("Frame not captured! ")


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



    cv2.imshow("ROI", gray)
    #cv2.imshow("Grayscale Image", gray)

cap.release()
cv2.destroyAllWindows()