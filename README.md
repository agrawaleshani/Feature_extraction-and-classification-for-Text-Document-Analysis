"# Feature_extraction-and-classification-for-Text-Document-Analysis" 

Optical Character Recognition of Text document based on HoG feature extraction method and KNN classifier is successfully implemented using impressive binarization techniques for preprocessing. A dataset of odhiya text images is collected from text documents. Linear contrast stretching and morphological opening is applied to adjust the contrast level of images and to minimize the intensity variations in the background. The HoG feature extraction is deployed to extract the gradient directional features providing essential shape information of the character samples. At the end, the KNN classifier uses extracted HoG feature vectors for classification producing promising results.

Implementation of the HoG descriptor algorithm is as follows:
1.	Divide the image into small connected regions called cells, and for each cell compute a histogram of gradient directions or edge orientations for the pixels within the cell.
2.	Discretize each cell into angular bins according to the gradient orientation.
3.	Each cell's pixel contributes weighted gradient to its corresponding angular bin.
4.	Groups of adjacent cells are considered as spatial regions called blocks. The grouping of cells into a block is the basis for grouping and normalization of histograms.
5.	Normalized group of histograms represents the block histogram. The set of these block histograms represents the descriptor.

K means 
It uses clustering approach which aims to partition n input features into some k classes in which each observation belongs to the cluster with nearest mean, serving as a prototype of the cluster.

1.	It starts with K as the input which is how many clusters you want to find. Place K centroids in random locations in your space.
2.	Now, using the euclidean distance between data points and centroids, assign each data point to the cluster which is close to it.
3.	Recalculate the cluster centers as a mean of data points assigned to it.
4.	Repeat 2 and 3 until no further changes occur.


This is unsupervised learning of document images.

RUN COMMANDS:
python HOG_FEATURE_EXTRAC.py --training images/dataset
python KMeans.py
python extract.py
python folders.py
python divide.py --training images/dataset

