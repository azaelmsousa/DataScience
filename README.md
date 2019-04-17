# DataScience
Project for the subject of Data Science in Health.

# Objective
The objective of this project is to study different approaches of data science in the context of the spread of dengue at the state of Minas Gerais, Brazil.

# Approaches
- `Onntology`
- `Machine Learning`
- `Complex Networks`
- `Statistics`

# Datasets
For the caracterization of Dengue, we had to explore many datasets to build one that contains all important features for this problem.

1) Info-Dengue: This dataset was used to determine the epidemic level of dengue
  https://info.dengue.mat.br/
  
2) IGBE: This dataset was used to extract important features of the observed cities, like PIB.
  https://www.ibge.gov.br/estatisticas/multidominio/meio-ambiente/9073-pesquisa-nacional-de-saneamento-basico.html?=&t=resultados
  https://ww2.ibge.gov.br/home/geociencias/geografia/redes_fluxos/gestao_do_territorio_2014/base.shtm
  
3) InMet: Contains information regarding temperature and climate changes of the cities. Since the mosquitos that spread the dengue fever increase their reproduction rate at certain temperatures, it is a good feature for the model.
  http://www.inmet.gov.br/portal/index.php?r=home2/index
  
# Softwares
- `Machine Learning`
--- ML/prepareData.py
   For a given starting week, ending week, starting year and ending year, it downloads the data from Info-Dengue and concatenate them in a csv file.
