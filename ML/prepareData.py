import pandas as pd
import numpy as np
import argparse

def getArgs():
    parser = argparse.ArgumentParser(description='Download Info Dengue Dataset.')
    parser.add_argument("start_week", action='store', type=int, help="Start week.")
    parser.add_argument("end_week", action='store', type=int, help="End week.")
    parser.add_argument("start_year", action='store', type=int, help="Start year.")
    parser.add_argument("end_year", action='store', type=int, help="End year.")
    parser.add_argument("out_path", action='store', type=str, help="Path to the output dir. Example: ../data/info_dengue/")
    return parser.parse_args()

if __name__ == "__main__":

	args = getArgs()

	start_week = args.start_week
	end_week = args.end_week
	start_year = args.start_year
	end_year = args.end_year

	url = 'https://info.dengue.mat.br/api/alertcity'

	code_mg = 31 #Codigo IBGE do estado de Minas Gerais - 31

	pd_cities = pd.read_csv("../data/code_cities.csv")
	cities_mg = pd_cities.values
	cities_mg = cities_mg[cities_mg[:,5]==code_mg] #Selecionar apenas cidades de MG
	cities = cities_mg[:,[0,1]]


	print("- Preparing Data")
	data = []
	for code,name in cities:
		print("--- Downloading Info Dengue Data for City "+str(name))
		search_filter = ('disease=dengue&geocode='+str(code)+'&format=csv&' +	
	    				'ew_start='+str(start_week)+'&ew_end='+str(end_week) +
	    				'50&ey_start='+str(start_year)+'&ey_end='+str(end_year))
		df = pd.read_csv('%s?%s' % (url,search_filter))
		headers = ','.join(list(df.columns.values))
		#print(df.shape)
		if (df.shape[0] > 0):
			#data.append(df.values)
			np.savetxt(args.out_path+"/"+str(name)+".csv",df,fmt='%s',delimiter=',',header=headers, comments='')

	#info_dengue = np.concatenate(data)

	#print("- Writing output file")
	#np.savetxt(args.out_path,info_dengue,fmt='%s',delimiter=',',header=headers, comments='')






