import numpy as np
import cv2
import os

cap = cv2.VideoCapture(0)
faceDetector = cv2.CascadeClassifier(os.path.join("haar-cascade-files","haarcascade_frontalface_default.xml"))
smileDetector = cv2.CascadeClassifier(os.path.join("haar-cascade-files","haarcascade_smile.xml"))
eyeDetector = cv2.CascadeClassifier(os.path.join("haar-cascade-files","haarcascade_eye.xml"))

while(True):
    # Capture frame-by-frame
    try:
        ret, frame = cap.read()
        if ret:
            # Face detector
            #cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            #roi = frame[y:y+h,x:x+w]
            
            # minNeighbors > siamo sicuri che sia una faccia per
            faces = faceDetector.detectMultiScale(frame,scaleFactor=1.5,minNeighbors=8,flags=cv2.CASCADE_DO_ROUGH_SEARCH | cv2.CASCADE_SCALE_IMAGE)
           
            for xf,yf,w,h in faces:
                roi = frame[yf:yf+h,xf:xf+w]

                cv2.rectangle(frame,(xf,yf),(xf+w,yf+h),(255,0,0),2) # BGR

                eyes = eyeDetector.detectMultiScale(roi,scaleFactor=1.05,minNeighbors=3,flags=cv2.CASCADE_DO_ROUGH_SEARCH | cv2.CASCADE_SCALE_IMAGE)
                for xe,ye,w,h in eyes:
                    xn = xf + xe
                    yn = yf + ye
                    cv2.rectangle(frame,(xn,yn),(xn+w,yn+h),(0,255,0),2) # BGR

                smiles = smileDetector.detectMultiScale(roi,scaleFactor=1.5,minNeighbors=15,flags=cv2.CASCADE_DO_ROUGH_SEARCH | cv2.CASCADE_SCALE_IMAGE)
                for xs,ys,w,h in smiles:
                    xn = xf + xs
                    yn = yf + ys
                    cv2.rectangle(frame,(xn,yn),(xn+w,yn+h),(0,0,255),2) # BGR
                    
                #roi  = cv2.resize(roi, (412,412))
                #cv2.imshow('roi',roi)

            frame = cv2.putText(frame,"Ciao", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,  1, (255, 0, 0) , 2, cv2.LINE_AA) 
            
            # Display the resulting frame
            cv2.imshow('frame',frame)
            #cv2.imshow("roi", roi)
            if cv2.waitKey(33) & 0xFF == ord('q'): #premi q per chiudere
                break

    except Exception as e:
        print(e)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()