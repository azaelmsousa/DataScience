import pandas as pd
import numpy as np
import os


path = "../data/inmet/climate/"
files = os.listdir(path)
print(files)

l = []
headers = []
for f in files:
	print(f)
	df = pd.read_csv(path+f,delimiter=",")
	l.append(df.values)
	headers = ','.join(list(df.columns.values))
	print(df)

data = np.concatenate(l)
data = data.astype('float64')
print(data.shape)
np.savetxt("all.csv",data,fmt='%.5e',delimiter=',',header=headers, comments='')
#np.savetxt("all.csv",data,fmt='%.18e',delimiter=',',header=headers, comments='')
