import cv2
import numpy as np
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    if cv2.waitKey(1) == ord('q'):
        break
    
    kernel = np.array(
        [[0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]]
    )
    filtered = cv2.filter2D(frame, -1, kernel)
    cv2.imshow("Filtered", filtered)
    cv2.waitKey(20)
    cv2.imshow("Non-Filtered Image", frame)
cap.release()
cv2.destroyAllWindows()
