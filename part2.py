import numpy as np
import cv2
import os

cap = cv2.VideoCapture(0)

#model=cv2.CascadeClassifier(os.path.join("haar-cascade-files","haarcascade_frontalface_default.xml"))
smile=cv2.CascadeClassifier(os.path.join("haar-cascade-files","haarcascade_smile.xml"))
#eye=cv2.CascadeClassifier(os.path.join("haar-cascade-files","haarcascade_eye.xml"))

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Face detector
    #cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    #roi = frame[y:y+h,x:x+w]
    #faces = model.detectMultiScale(frame,scaleFactor=1.5,minNeighbors=3,flags=cv2.CASCADE_DO_ROUGH_SEARCH | cv2.CASCADE_SCALE_IMAGE)
    faces = smile.detectMultiScale(frame,scaleFactor=1.5,minNeighbors=3,flags=cv2.CASCADE_DO_ROUGH_SEARCH | cv2.CASCADE_SCALE_IMAGE)
    #faces = eye.detectMultiScale(frame,scaleFactor=1.5,minNeighbors=3,flags=cv2.CASCADE_DO_ROUGH_SEARCH | cv2.CASCADE_SCALE_IMAGE)
    
    print(faces)
    for x,y,w,h in faces:
        print(x,y,w,h)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2) # blue BGR

    frame = cv2.putText(frame,"Ciao", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,  1, (255, 0, 0) , 2, cv2.LINE_AA) 

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()