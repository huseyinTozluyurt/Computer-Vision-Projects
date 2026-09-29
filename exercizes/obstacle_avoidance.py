import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera error.")
        break

    height, width, _ = frame.shape

    # Lower half of image
    roi = frame[height//2:height, 0:width]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    obstacle_detected = False

    for contour in contours:
        area = cv2.contourArea(contour)

        if area > 1000:
            x, y, w, h = cv2.boundingRect(contour)

            obstacle_center_x = x + w // 2
            frame_center_x = width // 2

            cv2.rectangle(roi, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if abs(obstacle_center_x - frame_center_x) < 100:
                obstacle_detected = True
                cv2.putText(
                    roi,
                    "Obstacle Ahead",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

    if obstacle_detected:
        decision = "STOP or TURN"
    else:
        decision = "FORWARD"

    cv2.putText(
        frame,
        f"Decision: {decision}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    cv2.imshow("Robot Vision", frame)
    cv2.imshow("Edges", edges)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()