import pandas as pd
import numpy as np
import argparse

def getArgs():
    parser = argparse.ArgumentParser(description='Download Info Dengue Dataset.')
    parser.add_argument("start_week", action='store', type=int, help="Start week.")
    parser.add_argument("end_week", action='store', type=int, help="End week.")
    parser.add_argument("start_year", action='store', type=int, help="Start year.")
    parser.add_argument("end_year", action='store', type=int, help="End year.")
    parser.add_argument("out_path", action='store', type=str, help="Path to the output csv file. Example: ../data/info_dengue.csv")
    return parser.parse_args()

if __name__ == "__main__":

	args = getArgs()

	start_week = args.start_week
	end_week = args.end_week
	start_year = args.start_year
	end_year = args.end_year

	url = 'https://info.dengue.mat.br/api/alertcity?'

	code_mg = 31 #Codigo IBGE do estado de Minas Gerais

	pd_cities = pd.read_csv("../data/cities.csv")
	cities_mg = pd_cities.values
	cities_mg = cities_mg[cities_mg[:,5]==code_mg]
	codes = cities_mg[:,0]

	print("- Preparing Data")
	data = []
	for code in codes:
		print("--- Downloading Info Dengue Data for City "+str(code))
		search_filter = ('geocode='+str(code)+'&disease=dengue&format=csv&' +
	    				'ew_start='+str(start_week)+'&ew_end='+str(end_week) +
	    				'50&ey_start='+str(start_year)+'2017&ey_end='+str(end_year))
		df = pd.read_csv("%s%s" % (url,search_filter))
		data.append(df.values)

	info_dengue = np.concatenate(data)

	headers = ','.join(list(df.columns.values))

	print("- Writing output file")
	np.savetxt(args.out_path,info_dengue,fmt='%s',delimiter=',',header=headers)






