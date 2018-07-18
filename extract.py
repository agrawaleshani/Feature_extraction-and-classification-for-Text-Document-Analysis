#creating a text file of all the predictions 
import pandas as pd 
import argparse
import fnmatch
data=pd.read_csv("kmprete.csv")
print(data.info())
print(len(data['0'].unique()))
X=data['0'].unique()

print(data['0'].unique())

with open('class.txt', 'w') as filehandle:  
    for listitem in X:
        filehandle.write('%s\n' % listitem)

#1203 unique aspect ratios are there

