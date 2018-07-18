
import numpy as np
import pandas as pd 

from matplotlib import pyplot as plt 
import pandas as pd
from sklearn.cluster import KMeans
csv_file='hog_featurete.csv'
X=pd.read_csv(csv_file)
X=np.array(X,'float32')
model=KMeans(n_clusters=120)
model.fit(X)
all_predictions=model.predict(X)
df=pd.DataFrame(all_predictions)
filepath='kmprete.csv'
df.to_csv(filepath,index=False)
print(model.labels_[::10])

df=pd.read_csv("kmprete.csv")
df1=pd.read_csv("labelste.csv")
df['labels']=df1
df.to_csv("kmprete.csv")