import pandas as pd
import numpy as np


url = 'https://info.dengue.mat.br/api/alertcity?'

code_mg = 31 #Código IBGE do estado de Minas Gerais

cities = pd.read_csv("../data/cities.csv").values
cities_mg = cities[cities[:,5]==code_mg]
codes = cities_mg[:,0]

for code in codes:
	print(code)

