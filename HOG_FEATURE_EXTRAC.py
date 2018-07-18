import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
import argparse
import pandas as pd
from collections import Counter
import os 
from glob import glob
from imutils import paths
import fnmatch
from sklearn.preprocessing import LabelEncoder
list_hog_fd=[]
dictionary={}
labels=[]
responses=[]
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--training", required=True,help="path to the training images")

args = vars(ap.parse_args())



for dirpath, dirs, files in os.walk(args["training"]):
    for filename in fnmatch.filter(files, '*.bmp'):
        #print (filename)
        keyVal = filename
        
        #keyVal=keyVal.split('.',1)[0]
        #keyVal=keyVal.split('.',1)[0][9:]
        '''keyVal=dirpath[dirpath.rindex('/')+1:]
        keyVal=keyVal.split('\\',1)[1][-2:]
        keyVal=str(int(keyVal)-1)'''
        

        im = cv2.imread(dirpath+"/"+filename)
        
        
        # Convert to grayscale and apply Gaussian filtering
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        
        im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
        #out = np.zeros(im.shape,np.uint8)
        # Threshold the image
        ret3,im_th= cv2.threshold(im_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #im_th = cv2.adaptiveThreshold(im_gray,255,1,1,11,2)
        #ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
        
        #im_th=cv2.dilate(im_th, (3, 3))
        im_th=cv2.resize(im_th,(32,32))
        roi_hog_fd = hog(im_th, orientations=10, pixels_per_cell=(8, 8), cells_per_block=(2, 2),transform_sqrt=True, visualise=False,feature_vector=True)
        print (len(roi_hog_fd))
        # Find contours in the image
        '''_, ctrs, hier = cv2.findContours(im_th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE )
        #findContours( threshold_output, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

        # Get rectangles contains each contour
        rects = [cv2.boundingRect(ctr) for ctr in ctrs]

        # For each rectangular region, calculate HOG features and predict
        # the digit using Linear SVM.
        
        for cnt in ctrs:
            if cv2.contourArea(cnt)>68:
                [x,y,w,h] = cv2.boundingRect(cnt)
                if  h>28:
                    cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),3)
                    roi = im_th[y:y+h,x:x+w]
                    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
                    roi = cv2.dilate(roi, (3, 3))
                    roi_hog_fd = hog(roi, orientations=12, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)'''
                       
        key = keyVal
        
        responses.append(key)
        #print (roi_hog_fd)
        #print (keyVal[0])
        #cv2.imshow('im',im)
        #cv2.waitKey(0)
        
        list_hog_fd.append(roi_hog_fd)
        
hog_features = np.array(list_hog_fd, 'float32')
labels = np.array(responses)
print (labels)
df=pd.DataFrame(labels)
filepath="labelste.csv"
df.to_csv(filepath,index=False)
df=pd.DataFrame(hog_features)
filepath='hog_featurete.csv'
df.to_csv(filepath,index=False)

