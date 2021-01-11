import os
import cv2
import numpy as np

'''
Requisiti: Python, Pip installati
pip3 install virtualenv
virtualenv --python=python3 env

se errore: cannot be loaded because running scripts is disabled on this system
Aprire powersheel e inserire
Set-ExecutionPolicy RemoteSigned 
'''

path_mask = os.path.join("dataset","mask")
names_mask = os.listdir(path_mask)
for name in names_mask:
    path = os.path.join(path_mask,name)
    im = cv2.imread(path)
    #im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR) 
    #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    #im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    size = 64
    channel = 3
    cv2.imshow("kl",im)
    im = cv2.resize(im,(size,size))  # colonne,righe
    im = im.reshape(size*size*channel)
    print(im.shape)
    im = im.reshape(size,size,channel)
    #im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) 
    cv2.imshow("klk",im)  # funziona solo in google colab cv2.imshow("window",im)
    cv2.waitKey(0)
    print(im.shape)
    break