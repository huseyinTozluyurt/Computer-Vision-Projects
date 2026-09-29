import cv2

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()   
    if not ret:
        print("Video not captured! ")
        break
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blurred_frame = cv2.GaussianBlur(gray, (5,5), 0)
    
    edges = cv2.Canny(blurred_frame, 50, 150)
    
    contours, _= cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 250:
            cv2.drawContours(frame, [contour], -1, (0,255,0), 2)

    cv2.imshow("Filter Contours", frame)
    cv2.imshow("Edges", edges)




cap.release()
cv2.destroyAllWindows()