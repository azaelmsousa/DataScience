import pandas as pd
import numpy as np
import os

path = "../data/inmet/"
files = os.listdir(path)
print(files)

for f in files:
	df = pd.read_csv(path+f,delimiter=";")
	name=f.split(".")[0]
	print(name)
	mean = df[["TempBulboSeco","TempBulboUmido","UmidadeRelativa","PressaoAtmEstacao","DirecaoVento","VelocidadeVentoNebulosidade"]].mean(axis=0, skipna=True).fillna(0)
	mean_clima = pd.DataFrame(mean).transpose()
	print(mean_clima)
	print("-"*50)
	mean_clima.to_csv(path+name+"_climate.csv")