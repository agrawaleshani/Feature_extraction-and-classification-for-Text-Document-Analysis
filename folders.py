#creating folders according to the predictions made by the kmeans

import os

with open('class.txt','rU') as x:
	for line in x:
		line=line.strip().split(':')
		filename="_".join([i for i in line])
		os.mkdir(filename)