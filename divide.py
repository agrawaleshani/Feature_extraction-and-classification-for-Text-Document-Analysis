#dividing images into folders according to the kmeans results


import pandas as pd 
import argparse
import fnmatch
import os
import cv2
data=pd.read_csv("kmprete.csv")

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--training", required=True,help="path to the training images")

args = vars(ap.parse_args())

for dirpath, dirs, files in os.walk(args["training"]):
    for filename in fnmatch.filter(files, '*.bmp'):
        l=[]
        #print (filename)
        keyVal = filename
        img=cv2.imread(dirpath+"/"+filename)
        #keyVal=keyVal.split('.',1)[0]
        #keyVal=keyVal.split('.',1)[0][9:]
        #keyVal=filename[filename.rindex('/')+1:]
        #keyVal=keyVal.split('\\',1)[1]
        #keyVal=str(int(keyVal)-1)
        print(keyVal)
        print(filename)
        X=data[data['labels']==keyVal]
        l=X.values
        print(l[0][1])
        #l[0][2]=l[0][2].strip().split(':')
        foldername=str(l[0][1])
        print(foldername)
        #path1="ccknresults"
        path=foldername
        path2=filename
        cv2.imwrite(os.path.join(path ,path2), img)
        cv2.waitKey(0)
