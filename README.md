# DataScience
Project for the subject of Data Science in Health.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/azaelmsousa/DataScience/master)

# Objective
The objective of this project is to study different approaches of data science in the context of the spread of dengue at the state of Minas Gerais, Brazil.

# Approaches
- `Ontology`
- `Machine Learning`
- `Complex Networks`
- `Correlation between ML and CN approaches`

# Datasets
For the caracterization of Dengue, we had to explore many datasets to build one that contains all important features for this problem.

1) *Info-Dengue*: This dataset was used to determine the epidemic level of dengue.
https://info.dengue.mat.br/
  
2) *IGBE*: This dataset was used to extract important features of the observed cities, like PIB.
https://www.ibge.gov.br/estatisticas/multidominio/meio-ambiente/9073-pesquisa-nacional-de-saneamento-basico.html?=&t=resultados
https://ww2.ibge.gov.br/home/geociencias/geografia/redes_fluxos/gestao_do_territorio_2014/base.shtm
  
3) *InMet*: Contains information regarding temperature and climate changes of the cities. Since the mosquitos that spread the dengue fever increase their reproduction rate at certain temperatures, it is a good feature for the model.
http://www.inmet.gov.br/portal/index.php?r=home2/index

4) *Epidemic Report - State Secretary of Health - Minas Gerais*: Contains informations regarding dengue cases among the state of Minas Gerais during the years of 2017, 2018 and 2019. The dataset contains data such as incidence level, amount of cases per month, city situation, etc. It is a good and real-time source of dengue information, which can be used for both Machine Learning and Complex Networks.

5) *Aedes Aegypti Quick Index Survey - - State Secretary of Health - Minas Gerais*: COntains information regarding waste disposal in each municipality of the state, in order to target mosquitoes targets. This feature is good to be used for Machine Learning.
  
# Softwares
- `Machine Learning`
--- ML/prepareData.py
   For a given starting week, ending week, starting year and ending year, it downloads the data from Info-Dengue and concatenate them in a csv file.
   
- `Complex Network`
--- Gephi 0.9.2 (https://gephi.org/users/download/)
 
