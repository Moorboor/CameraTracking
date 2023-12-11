import cv2

cap = cv2.VideoCapture(2) 
# cap.set(cv2.CAP_PROP_FRAME_WIDTH , 800) 
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT , 600) 



ret, frame_source = cap.read()

print(ret)
while True:

    ret, frame_source = cap.read()
    if not ret:
        break

    cv2.imshow("Original", frame_source)

    # Quit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break 



# Release the VideoCapture object
cap.release()
# Close all OpenCV windows
cv2.destroyAllWindows()