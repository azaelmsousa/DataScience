import numpy as np
import pandas as pd

'''
Dicionário de Variáveis:

VAR01	Hierarquia urbana do município A do par de ligação segundo REGIC 2007		IBGE	2016	-
VAR02	Hierarquia urbana do município B do par de ligação segundo REGIC 2007		IBGE	2016	-
VAR03	Custo mínimo do par de ligação                                      	    IBGE	2016	R$
VAR04	Tempo mínimo de deslocamento do par de ligação	               		        IBGE	2016	Minutos
VAR05	Frequência de saídas de veículos hidroviários no par de ligação				IBGE	2016	Número de saídas
VAR06	Frequência de saídas de veículos rodoviários no par de ligação				IBGE	2016	Número de saídas
VAR07	Frequência total de saídas de veículos no par de ligação					IBGE	2016	Número de saídas
VAR08	Longitude da sede municipal A do par de ligação								IBGE	2016	Graus decimais
VAR09	Latitude da sede municipal A do par de ligação								IBGE	2016	Graus decimais
VAR10	Longitude da sede municipal B do par de ligação								IBGE	2016	Graus decimais
VAR11	Latitude da sede municipal B do par de ligação								IBGE	2016	Graus decimais
VAR12	Frequência de saídas de veículos que não declaram CNPJ no par de ligação	IBGE	2016	Número de saídas
VAR13	Marcas de imputação. Existência de seção imputada no par de ligação			IBGE	2016	-
VAR14	Custo relativo: custo / tempo												IBGE	2016	R$/minuto
'''


print("--- Loading Data")
df = pd.read_csv('../data/ligacoes_rodoviarias_e_hidroviarias_2016.csv')

print("--- Selecting Important Columns")
df_np = df.values
df_np = df_np[df_np[:,1] == 31] # uf da cidade origem é MG
df_np = df_np[df_np[:,5] == 31] # uf da cidade destino é MG
df_np = df_np[:,[4,8,11,12,15]]
'''
Column 4 - cidade origem
Column 8 - cidade destino
Column 11 - VAR3
Column 11 - VAR4
Column 15 - VAR7
'''

print("--- Preparing data")
cost = df_np[:,[0,1,2]]
time = df_np[:,[0,1,3]]
freq = df_np[:,[0,1,4]]

print("--- Output data")
header = 'source,target,weight'
np.savetxt("../data/cost.csv",cost,fmt='%s',delimiter=',',header=header,comments='')
np.savetxt("../data/time.csv",time,fmt='%s',delimiter=',',header=header,comments='')
np.savetxt("../data/freq.csv",freq,fmt='%s',delimiter=',',header=header,comments='')


