import cv2
import numpy as np
import pickle
import joblib

'''
Spiegare
problema con versioni sklearn (colab vs python)
'''

def load_model_joblib(inp_name):
    # Load a pipeline
    my_model_loaded = joblib.load(inp_name)
    return my_model_loaded

class FaceMaskDetector:
	
    def __init__(self):
        self.model_face = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
        self.model_mask = load_model_joblib("models/model_pickle.pkl")
        self.size = 64
        self.channel = 3

    def  findLargestBB(self,bbs):
      areas = [w*h for x,y,w,h in bbs]
      if not areas:
          return False, None
      else:
          i_biggest = np.argmax(areas) 
          biggest = bbs[i_biggest]
          return True, biggest

    def run(self, image):
        image_proc = image
        if self.channel==1:
            image_proc = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.model_face.detectMultiScale(image,scaleFactor=1.1,minNeighbors=4, flags=cv2.CASCADE_DO_ROUGH_SEARCH | cv2.CASCADE_SCALE_IMAGE)
        isFaceDetected, face = self.findLargestBB(faces)
        if isFaceDetected:
          for (x,y,w,h) in [face]:
          #for (x,y,w,h) in faces: 
              #print("Face found")
              roi = image_proc[y:y+h,x:x+w]
              # HA oppure non  ha la mascherina
              roi_resized = cv2.resize(roi,( self.size , self.size ))
              #print(roi_resized.shape)
              #roi_resized = roi_resized[np.newaxis,:,:,:]
              vec = roi_resized.reshape(1,self.size*self.size*self.channel)
              #print(vec.shape)
              res = self.model_mask.predict_proba(vec)  
              #print(res.shape)          
              mask_perc = res[0][0]
              #no_mask_perc = res[0][1]
              #print( "Mask: " + str(np.round(mask_perc,1)) )
              cv2.putText(image,"Mask: " + str(np.round(mask_perc,1)), (10,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
              #cv2.imwrite(self.save_folder+str(self.c)+".png", roi)
              cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        return image
   
fm = FaceMaskDetector()
cap = cv2.VideoCapture(1)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret:
        # Our operations on the frame come here
        frame = fm.run(frame)
    
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()