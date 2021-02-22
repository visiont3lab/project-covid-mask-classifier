from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import SpectralClustering, KMeans
import cv2
import numpy as np
import plotly.graph_objects as go

# LINK: https://visiont3lab.github.io/tecnologie_data_science/docs/unsupervised/clustering.html

'''
pip install plotly==4.14.0 opencv-python scikit-learn pillow
'''

# -- Load Image --> Load sample image load images as RGB
im = load_sample_image('flower.jpg') # china.jpg
#im = cv2.imread("replace_this_with_image_path", 1) # 1 --> color, 0 --> gray

# -- Covert loaded image to BGR
im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

# -- Resize Image
im = cv2.resize(im, (256,256)) #w, h

# -- Visualize Initial Image (press enter to move on)
cv2.imshow("Initial Image", im)
#cv2.waitKey(0)

# -- Get Image size
w, h, d = tuple(im.shape)
print("Width: %s , Height: %s , Depth: %s" % (w,h,d))

# -- Flatten Image
im_flat = np.reshape(im, (w * h, d))
print("Image Flat shape: ", im_flat.shape)

# -- Scaling 0-1
im_flat = np.array(im_flat, dtype=np.float64) / 255
# or min max scaler
#mms = MinMaxScaler(feature_range=(0,1))
#im_flat = mms.fit_transform(im_flat)

# -- Apply Kmeans
print("--- Calculate Centers and Cluster Image")
kmeans = KMeans(n_clusters=2,max_iter=300,tol=0.0001)
#kmeans = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', assign_labels='kmeans')
kmeans.fit(im_flat)
labels = kmeans.predict(im_flat) #hard clustering
centers = kmeans.cluster_centers_
print(centers)

# Get image
im_kmeans = np.zeros(im_flat.shape, dtype=np.uint8)
for i in range(0, im_flat.shape[0]):
    lb = labels[i]
    c = centers[lb] # 0,1
    c = np.array(c*255,dtype=np.uint8)
    im_kmeans[i,:] = c
im_kmeans = im_kmeans.reshape((w,h,d))
cv2.imshow("Cluster Image", im_kmeans)
#cv2.waitKey(0)

# Elbow Method analysis
print("--- Elbow method analysis")
inertias = [] 
K = [2,3,4,5,6,7,8]
for k in K: 
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(im_flat)       
    labels = kmeans.predict(im_flat) #hard clustering
    centers = kmeans.cluster_centers_
    
    im_kmeans = np.zeros(im_flat.shape, dtype=np.uint8)
    for i in range(0, im_flat.shape[0]):
        lb = labels[i]
        c = centers[lb] # 0,1
        c = np.array(c*255,dtype=np.int)
        im_kmeans[i,:] = c
    im_kmeans = im_kmeans.reshape((w,h,d))

    #im_new = cv2.cvtColor(im_new, cv2.COLOR_BGR2RGB)
    cv2.imshow("Kmeans Cluster %s" % (k),im_kmeans)
    #cv2.imwrite("im_"+str(k)+".png",im_kmeans)
    #cv2.waitKey(0)
    inertias.append(kmeans.inertia_) 

fig = go.Figure()
fig.add_trace(go.Scatter(
    x = K,
    y = inertias, 
    mode="lines+markers",
    text = labels, 
    )
)
fig.update_layout(
    title="The Elbow Method using Inertia",
    xaxis_title="Values of K",
    yaxis_title="Inertia",
    hovermode="x"
)
#fig.write_html("Elbow.html")
fig.show()