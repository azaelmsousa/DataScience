# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os

df_freq = pd.read_csv("../data/freq.csv")
np_freq = df_freq.values
print(np_freq)

files = os.listdir("../data/info_dengue")
print(files)

data = []
for f in files:
	basename = os.path.splitext(f)[0]
	print(basename)
	file = open(u"../data/info_dengue/"+f)
	df_city = pd.read_csv(file)
	file.close()
	try:
		freq1 = np.sum(np_freq[np_freq[:,1] == basename][:,2])
		print(freq1)
	except:
		print("error1")
		freq1 = 0
	try:
		freq2 = np.sum(np_freq[np_freq[:,0] == basename][:,2])
		print(freq2)
	except:
		print("error")
		freq2 = 0
	df_city.insert(5, "freq", int(freq1+freq2))
	headers = ','.join(list(df_city.columns.values[[6,7,8,10]]))
	df = df_city.values[:,[6,7,8,10]]
	data.append(df)
	

np_data = np.vstack(data)
print(np_data)
np.savetxt("../data/info_dengue_variables.csv",np_data,fmt='%d',delimiter=',',header=headers, comments='')

