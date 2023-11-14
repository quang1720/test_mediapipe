import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('image.jpg', frame)
        break

cap.release()
cv2.destroyAllWindows()
