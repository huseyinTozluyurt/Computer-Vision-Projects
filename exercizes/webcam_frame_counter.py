import cv2
cap = cv2.VideoCapture(0)

frame_counter = 0

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (1024, 800))

    if ret:
        frame_counter += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    # print("Frame Counter: " + str(frame_counter))
    cv2.putText(frame, f'Frame Counter: {frame_counter}', (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)
    cv2.imshow("Frame", frame)


cap.release()

cv2.destroyAllWindows()