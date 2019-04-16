import pandas as pd
import numpy as np


url = 'https://info.dengue.mat.br/api/alertcity?'

code_mg = 31 #Código IBGE do estado de Minas Gerais

pd_cities = pd.read_csv("../data/cities.csv")
cities_mg = pd_cities.values
cities_mg = cities_mg[cities_mg[:,5]==code_mg]
codes = cities_mg[:,0]

print("- Preparing Data")
data = []
for code in codes:
	print("--- Downloading Info Dengue Data for City "+str(code))
	search_filter = ('geocode='+str(code)+'&disease=dengue&format=csv&' +
    				'ew_start=1&ew_end=50&ey_start=2017&ey_end=2017')
	df = pd.read_csv("%s%s" % (url,search_filter)).values
	data.append(df)

info_dengue = np.concatenate(data)

headers = ','.join(list(pd_cities.columns.values))
np.savetxt("../data/info_dengue_mg.csv",info_dengue,fmt='%s',delimiter=',',header=headers)






