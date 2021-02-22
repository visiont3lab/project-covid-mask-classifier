import cv2
import os
import joblib
import numpy as np 
import onnxruntime as rt

model=cv2.CascadeClassifier(os.path.join("haar-cascade-files","haarcascade_frontalface_default.xml"))

# Pickle
#modelsvm = joblib.load(os.path.join("models","model_pickle_augmented.pkl"))

# Onnx
model_onnx =  rt.InferenceSession(os.path.join("models","model_onnx_augmented.onnx"))
input_name = model_onnx.get_inputs()[0].name
label_name = model_onnx.get_outputs()[1].name  # 0 output_label, 1 output_proability

cap = cv2.VideoCapture(0) # Webcam --> 0,1

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret:
        # Our operations on the frame come here
        #frame=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        #frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        # all'aumentare velocizza l'agoritmo
        # confidence --> all'aumentare è piu sicuro
        faces = model.detectMultiScale(frame,scaleFactor=1.5,minNeighbors=5,flags=cv2.CASCADE_DO_ROUGH_SEARCH | cv2.CASCADE_SCALE_IMAGE)
        for x,y,w,h in faces:
            # x --> colonne
            # y --> righe
            pt1 = (x,y) # upperleft corner
            pt2 = (x+w,y+h) # bottom right corner

            # Roi
            roi=frame[y:y+h,x:x+w] # Righe, colonne
           
            # Utilizziamo la roi nel svm classifier
            roi_resized = cv2.resize(roi,( 64 , 64 )) # (64,64,3)
            cv2.imshow("Roi",roi_resized)
            vec = roi_resized.reshape(1,64*64*3) 
            
            # Onnx
            res = model_onnx.run([label_name], {input_name: vec.astype(np.float32)})[0] # return a list of dick
            
            # Pickle
            #res = modelsvm.predict_proba(vec)  
      
            mask_perc =res[0][0]
            nomask_perc =res[0][1]
            
            # Color
            cv2.rectangle(frame,pt1,pt2,(255,0,0),2) # blue BGR
            write="Mask: " + str( np.round(mask_perc,1) )
            cv2.putText(frame,write, (10,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
            
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()