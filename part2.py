import numpy as np
import cv2
import os
import joblib 

cap = cv2.VideoCapture(1)
faceDetector = cv2.CascadeClassifier(os.path.join("haar-cascade-files","haarcascade_frontalface_default.xml"))
smileDetector = cv2.CascadeClassifier(os.path.join("haar-cascade-files","haarcascade_smile.xml"))
eyeDetector = cv2.CascadeClassifier(os.path.join("haar-cascade-files","haarcascade_eye_tree_eyeglasses.xml"))
maskClassifier = joblib.load("models/model_pickle_augmented.pkl")

while(True):
    # Capture frame-by-frame
    try:
        ret, frame = cap.read()
        if ret:
            # minNeighbors > siamo sicuri che sia una faccia per
            faces = faceDetector.detectMultiScale(frame,scaleFactor=1.5,minNeighbors=8,flags=cv2.CASCADE_DO_ROUGH_SEARCH | cv2.CASCADE_SCALE_IMAGE)

            for xf,yf,wf,hf in faces:
                roi = frame[yf:yf+hf,xf:xf+wf]

                roiBlur = cv2.medianBlur(roi, 15)
                roiHSV = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                frame[yf:yf+hf,xf:xf+wf] = roiBlur

                cv2.rectangle(frame,(xf,yf),(xf+wf,yf+hf),(255,0,0),2) # BGR

                eyes = eyeDetector.detectMultiScale(roi,scaleFactor=1.05,minNeighbors=3,flags=cv2.CASCADE_DO_ROUGH_SEARCH | cv2.CASCADE_SCALE_IMAGE)
                for xe,ye,we,he in eyes:
                    xn = xf +xe
                    yn = yf + ye
                    cv2.rectangle(frame,(xn,yn),(xn+we,yn+he),(0,255,0),2) # BGR

                smiles = smileDetector.detectMultiScale(roi,scaleFactor=1.5,minNeighbors=15,flags=cv2.CASCADE_DO_ROUGH_SEARCH | cv2.CASCADE_SCALE_IMAGE)
                for xs,ys,ws,hs in smiles:
                    xn = xf +xs
                    yn = yf + ys
                    cv2.rectangle(frame,(xn,yn),(xn+ws,yn+hs),(0,0,255),2) # BGR

                # Mask detector
                size_training = 64
                im = cv2.resize(roi, (size_training,size_training))
                im = im.flatten().reshape(1,-1)
                pred = maskClassifier.predict_proba(im) # 1x64x64x3
                proba_mask = pred[0][0]
                proba_nomask = pred[0][1]
                frame = cv2.putText(frame,("Mask: " + str(np.round(proba_mask,2))), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,  1, (255, 0, 0) , 2, cv2.LINE_AA) 

            frame = cv2.putText(frame,"Ciao", (50, 150), cv2.FONT_HERSHEY_SIMPLEX,  1, (255, 0, 0) , 2, cv2.LINE_AA) 
            
            # Display the resulting frame
            cv2.imshow('frame',frame)
            #cv2.imshow("roi", roi)
            if cv2.waitKey(33) & 0xFF == ord('p'):
                break

    except Exception as e:
        print(e)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()