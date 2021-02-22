import cv2
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, precision_score, recall_score,accuracy_score
import pandas as pd

def save_model_pickle(inp_name,inp_clf):
    #https://stackoverflow.com/questions/10592605/save-classifier-to-disk-in-scikit-learn
    with open(inp_name, 'wb') as f:
        pickle.dump(inp_clf, f) 

def save_model_joblib(inp_name,inp_clf):
    joblib.dump(inp_clf, inp_name) 

def load_model_joblib(inp_name):
    # Load a pipeline
    my_model_loaded = joblib.load(inp_name)
    return my_model_loaded

def load_model_pickle(inp_name):
    with open(inp_name, 'rb') as f:
        model = pickle.load(f)
    return model


folders = ["dataset/mask","dataset/no-mask"]
class_index = [0,1]
size_training = 64
channel = 3
X = []
Y = []
for folder, idx in zip(folders, class_index):
    names = sorted(os.listdir(folder))
    for name in names:
        #path = folder + "/" + name 
        path = os.path.join(folder,name )
        im = cv2.imread(path,1) # 1
        im = cv2.resize(im, (size_training,size_training))
        #print(im.shape)
        im = im.reshape((size_training*size_training*channel))
        #im = im.flatten() #.reshape(1,-1)
        X.append(im)
        Y.append(idx)
        #print(im.shape)
        #im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        #im_gray = cv2.resize(im_gray, (size_training,size_training))
        #cv2_imshow(im) # cv2.imshow("im", im)
        #cv2_imshow(im_gray) # cv2.imshow("im", im)
# -- X,Y
X = np.array(X, dtype=np.float)
Y = np.array(Y, dtype=np.float)
print(X.shape)
print(Y.shape)

# Train test
X_train, X_test, Y_train, Y_test  = train_test_split(X, Y, test_size=0.1, random_state=42, shuffle=True)

pipeline = Pipeline([
    ("sc", MinMaxScaler(feature_range=(0,1))), # 0-1 features
    #("pca", PCA(n_components=0.99)),
    ("model", SVC(probability=True) ) # Probability truw slow down dataset
])

pipeline.fit(X_train,Y_train)
score_train = pipeline.score(X_train,Y_train)
score_test = pipeline.score(X_test,Y_test)
print("f1_weighted score Train: ", score_train)
print("f1_weighted score Test: ", score_test)

save_model_joblib("model_pickle.pkl", pipeline)

Y_pred_test = pipeline.predict(X_test)
cm = confusion_matrix(Y_test,Y_pred_test)
df = pd.DataFrame(cm, columns=["Pred-Mask","Pred-No-Mask"], index=["Real-Mask","Real-No-Mask"])
print(df)

precision = precision_score(Y_test,Y_pred_test)
recall = recall_score(Y_test,Y_pred_test)
accuracy = accuracy_score(Y_test,Y_pred_test)
F1Score = 2* (precision*recall) / (precision + recall) # (Precision+Recall)/2
print(recall)
print(precision)
print(accuracy)
print(F1Score)

# USARE IL Modello
model = load_model_joblib("model_pickle.pkl")
path = "dataset/mask/download.jpg"
im = cv2.imread(path,1) # 1
size_training = 64
im = cv2.resize(im, (size_training,size_training))
im = im.flatten().reshape(1,-1)
print(im.shape)
Y_pred = model.predict(im) # 1x64x64x3
print(Y_pred)