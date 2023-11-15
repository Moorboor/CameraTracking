import cv2


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
cap.set(cv2.CAP_PROP_FRAME_WIDTH , 320) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT , 240) 


# Define the codec and create a VideoWriter object to save the video in MP4 format
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec for MP4 format
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (320, 240))  # 'output.mp4' is the output file name




while True:
    success,img = cap.read()
    cv2.imshow('Live cam    ',img)

    out.write(img)   

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

