import numpy as np
import cv2
import joblib

model = joblib.load("model_pickle.pkl")
size_training = 64

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Face detector
    x = 260
    y = 190
    w = 160
    h = 170
    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    roi = frame[y:y+h,x:x+w]
    
    # Mask detector
    im = cv2.resize(roi, (size_training,size_training))
    im = im.flatten().reshape(1,-1)
    pred = model.predict_proba(im) # 1x64x64x3
    proba_mask = pred[0][0]
    proba_nomask = pred[0][1]
    #pred = model.predict(im) # 1x64x64x3
    #print("Mask: ", proba_mask)
    #print("No Mask: ", proba_nomask)

    frame = cv2.putText(frame,("Mask: " + str(np.round(proba_mask,2))), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,  1, (255, 0, 0) , 2, cv2.LINE_AA) 
    
    # Display the resulting frame
    cv2.imshow('frame',frame)
    #cv2.imshow("roi", roi)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()