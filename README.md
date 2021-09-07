# Analise Exploratoria e Previsao de Arremessos da NBA

# Objetivo


* Este projeto visa primeiramente, realizar uma análise exploratória dos dados obtidos das últimas 6 temporadas regulares da NBA (2015-16 a 2020-21) e treinar diferentes modelos de machine learning com o intuito de prever se um arremesso é bem-sucedido ou não.
* Os dados foram obtidos através da API da NBA, o script 'get_players_shot_charts.ipynb' criado e a planilha com os ID's dos jogadores pode ser encontrados em: 

- (https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA) 

# Linguagem, Bibliotecas e Pacotes
    
O trabalho foi feito todo em Python 3. Abaixo, segue a listagem de todas bibliotecas e pacotes utilizados:

    #Import Libs

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import missingno as msno
    from pandas_profiling import ProfileReport
    import plotly.express as px
    import matplotlib as mpl
    import time
    from matplotlib.patches import Circle, Rectangle, Arc, ConnectionPatch
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    
    # Import Libs

    from sklearn.feature_selection import VarianceThreshold
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.svm import LinearSVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import tree
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import classification_report
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    import xgboost as xgb
    
# Leitura dos Dados

As 6 planilhas foram importandas e inseridas em Dataframes utilizando a biblioteca pandas.

    #Read NBA Shots excel files

    nba_shots_2020_21 = pd.read_excel('nba_shots_2020-21.xlsx', engine='openpyxl')
    nba_shots_2019_20 = pd.read_excel('nba_shots_2019-20.xlsx', engine='openpyxl')
    nba_shots_2018_19 = pd.read_excel('nba_shots_2018-19.xlsx', engine='openpyxl')
    nba_shots_2017_18 = pd.read_excel('nba_shots_2017-18.xlsx', engine='openpyxl')
    nba_shots_2016_17 = pd.read_excel('nba_shots_2016-17.xlsx', engine='openpyxl')
    nba_shots_2015_16 = pd.read_excel('nba_shots_2015-16.xlsx', engine='openpyxl')
    
Cada Dataframe tem a coluna "Unnamed: 0" retirada e a coluna "SEASON_ID" adicionada sendo inserida a respectiva temporada do Dataframe em questão.

    #Drop "Unnamed: 0" column, Add "SEASON_ID" column in nba_shots_2020_21

    nba_shots_2020_21.drop(['Unnamed: 0'], axis=1, inplace=True)
    nba_shots_2020_21['SEASON_ID'] = '2020-21'
    nba_shots_2020_21.head()

    #Drop "Unnamed: 0" column, Add "SEASON_ID" column in nba_shots_2019_20

    nba_shots_2019_20.drop(['Unnamed: 0'], axis=1, inplace=True)
    nba_shots_2019_20['SEASON_ID'] = '2019-20'
    nba_shots_2019_20.head()

    #Drop "Unnamed: 0" column, Add "SEASON_ID" column in nba_shots_2018_19

    nba_shots_2018_19.drop(['Unnamed: 0'], axis=1, inplace=True)
    nba_shots_2018_19['SEASON_ID'] = '2018-19'
    nba_shots_2018_19.head()

    #Drop "Unnamed: 0" column, Add "SEASON_ID" column in nba_shots_2017_18

    nba_shots_2017_18.drop(['Unnamed: 0'], axis=1, inplace=True)
    nba_shots_2017_18['SEASON_ID'] = '2017-18'
    nba_shots_2017_18.head()

    #Drop "Unnamed: 0" column, Add "SEASON_ID" column in nba_shots_2016_17

    nba_shots_2016_17.drop(['Unnamed: 0'], axis=1, inplace=True)
    nba_shots_2016_17['SEASON_ID'] = '2016-17'
    nba_shots_2016_17.head()

    #Drop "Unnamed: 0" column, Add "SEASON_ID" column in nba_shots_2015_16

    nba_shots_2015_16.drop(['Unnamed: 0'], axis=1, inplace=True)
    nba_shots_2015_16['SEASON_ID'] = '2015-16'
    nba_shots_2015_16.head()

Os 6 Dataframes são concatenados em um único novo Dataframe chamado 'nba_shots'.

    #Create nba_shots as a concatenation of the 3 Dataframes from each reagular season

    nba_shots = pd.concat([nba_shots_2020_21, nba_shots_2019_20, nba_shots_2018_19, nba_shots_2017_18, nba_shots_2016_17, nba_shots_2015_16], sort=False)

    nba_shots.head()


# Análise Inicial de nba_shots

    #Get nba_shots dataframe shape

    nba_shots.shape
    
    (989617, 25)
    
    #Get nba_shots dataframe columns

    nba_shots.columns
    
    Index(['GRID_TYPE', 'GAME_ID', 'GAME_EVENT_ID', 'PLAYER_ID', 'PLAYER_NAME',
       'TEAM_ID', 'TEAM_NAME', 'PERIOD', 'MINUTES_REMAINING',
       'SECONDS_REMAINING', 'EVENT_TYPE', 'ACTION_TYPE', 'SHOT_TYPE',
       'SHOT_ZONE_BASIC', 'SHOT_ZONE_AREA', 'SHOT_ZONE_RANGE', 'SHOT_DISTANCE',
       'LOC_X', 'LOC_Y', 'SHOT_ATTEMPTED_FLAG', 'SHOT_MADE_FLAG', 'GAME_DATE',
       'HTM', 'VTM', 'SEASON_ID'],
      dtype='object')
      
      #Get nba_shots dataframe describe

      nba_shots.describe()
      
      	GAME_ID	GAME_EVENT_ID	PLAYER_ID	TEAM_ID	PERIOD	MINUTES_REMAINING	SECONDS_REMAINING	SHOT_DISTANCE	LOC_X	LOC_Y	SHOT_ATTEMPTED_FLAG	SHOT_MADE_FLAG	GAME_DATE
    count	9.896170e+05	989617.000000	9.896170e+05	9.896170e+05	989617.000000	989617.000000	989617.000000	989617.000000	989617.000000	989617.000000	989617.0	989617.000000	9.896170e+05
    mean	2.177412e+07	303.451679	6.966109e+05	1.610613e+09	2.467222	5.350666	28.862704	12.999324	-1.319433	92.180580	1.0	0.463385	2.018430e+07
    std	1.661812e+05	189.289854	6.850221e+05	8.705219e+00	1.134207	3.429668	17.431075	10.434087	108.356158	94.014776	0.0	0.498658	1.789421e+04
    min	2.150000e+07	2.000000	2.544000e+03	1.610613e+09	1.000000	0.000000	0.000000	0.000000	-250.000000	-52.000000	1.0	0.000000	2.015103e+07
    25%	2.160104e+07	138.000000	2.023390e+05	1.610613e+09	1.000000	2.000000	14.000000	2.000000	-46.000000	11.000000	1.0	0.000000	2.017032e+07
    50%	2.180042e+07	298.000000	2.035000e+05	1.610613e+09	2.000000	5.000000	29.000000	12.000000	0.000000	51.000000	1.0	0.000000	2.018121e+07
    75%	2.190073e+07	454.000000	1.627741e+06	1.610613e+09	3.000000	8.000000	44.000000	24.000000	43.000000	175.000000	1.0	1.000000	2.020020e+07
    max	2.200108e+07	1012.000000	1.630466e+06	1.610613e+09	8.000000	12.000000	59.000000	87.000000	250.000000	867.000000	1.0	1.000000	2.021052e+07
    
    #Get nba_shots dataframe info

    nba_shots.info()
    
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 989617 entries, 0 to 123985
    Data columns (total 25 columns):
     #   Column               Non-Null Count   Dtype 
    ---  ------               --------------   ----- 
     0   GRID_TYPE            989617 non-null  object
     1   GAME_ID              989617 non-null  int64 
     2   GAME_EVENT_ID        989617 non-null  int64 
     3   PLAYER_ID            989617 non-null  int64 
     4   PLAYER_NAME          989583 non-null  object
     5   TEAM_ID              989617 non-null  int64 
     6   TEAM_NAME            989617 non-null  object
     7   PERIOD               989617 non-null  int64 
     8   MINUTES_REMAINING    989617 non-null  int64 
     9   SECONDS_REMAINING    989617 non-null  int64 
     10  EVENT_TYPE           989617 non-null  object
     11  ACTION_TYPE          989617 non-null  object
     12  SHOT_TYPE            989617 non-null  object
     13  SHOT_ZONE_BASIC      989617 non-null  object
     14  SHOT_ZONE_AREA       989617 non-null  object
     15  SHOT_ZONE_RANGE      989617 non-null  object
     16  SHOT_DISTANCE        989617 non-null  int64 
     17  LOC_X                989617 non-null  int64 
     18  LOC_Y                989617 non-null  int64 
     19  SHOT_ATTEMPTED_FLAG  989617 non-null  int64 
     20  SHOT_MADE_FLAG       989617 non-null  int64 
     21  GAME_DATE            989617 non-null  int64 
     22  HTM                  989617 non-null  object
     23  VTM                  989617 non-null  object
     24  SEASON_ID            989617 non-null  object
    dtypes: int64(13), object(12)
    memory usage: 216.3+ MB

# Checagem de valores nulos

nba_shots não possui nenhum valor faltante.

Foi utilizado a biblioteca missingno para realizar a checagem.

![Missingno_plot](https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/Images/missingno_plot.png)
    
    
# Relatório Pandas Profile

Foi gerado o 'Pandas Profile Report' que oferece uma análise extensa do conjunto de dados que está sendo abordado.

- (https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/nba_shots_pandas_profile_report.html)
    

# Função para desenhar a quadra

A função 'create_court' abaixo foi obtida do seguinte artigo: 
    
- (https://towardsdatascience.com/make-a-simple-nba-shot-chart-with-python-e5d70db45d0d)

Esta função cria desenha uma quadra de basquete nas proporções da NBA utilizando matplotlib.
    

# Análise Exploratória


# 1. Arremessos por Jogador

De ínicio foram plotados todos os arremessos tentados na temporada 2020-21 de 3 atletas da liga (James Harden, Stephen Curry e Nikola Jokic). 
 
![james_harden_2020-21_shot_chart](https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/Images/james_harden_2020-21_shot_chart.png)

![stephen_curry_2020-21_shot_chart](https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/Images/stephen_curry_2020-21_shot_chart.png)

![nikola_jokic_2020-21_shot_chart](https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/Images/nikola_jokic_2020-21_shot_chart.png)

# 2. Arremessos Acertados por Temporada

Abaixo temos as plotagens utilizando a função .hexbin da biblioteca matplotlib.

Nessa sequência de gráficos possível notar como o arremesso de 3 pontos se tornou cada vez mais o arremesso* mais popular na liga.

*Arremessos não incluem ações ofensivas como bandejas e enterradas, que são feitas próximas da cesta e que continuam proeminentes na liga como pode ser notado em todas as imagens.

![2015-16_regular_season_made_shots](https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/Images/2015-16_regular_season_made_shots.png)

![2016-17_regular_season_made_shots](https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/Images/2016-17_regular_season_made_shots.png)

![2017-18_regular_season_made_shots](https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/Images/2017-18_regular_season_made_shots.png)

![2018-19_regular_season_made_shots](https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/Images/2018-19_regular_season_made_shots.png)

![2019-20_regular_season_made_shots](https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/Images/2019-20_regular_season_made_shots.png)

![2020-21_regular_season_made_shots](https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/Images/2020-21_regular_season_made_shots.png)
    

# 3. Outras Visualizações de Arremessos

Os dados obtidos permitem ainda outras plotagens dos arremessos a seguir são mostradas 3 diferentes formas de enxergar os arremessos de acordo com sua posição em quadra.
    

Arremessos acertados por região da quadra

O atributo 'SHOT_ZONE_AREA' oferece as regiões da quadra utilizadas nessa plotagem.

![2020-21_regular_season_shots_made_per_zone_area](https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/Images/2020-21_regular_season_shots_made_per_zone_area.png)
    
Arremessos acertados por zonas de distância.

O atributo 'SHOT_ZONE_RANGE' oferece as zonas por diferentes distâncias utilizadas nessa pltagem.

![2020-21_regular_season_shots_made_per_zone_range](https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/Images/2020-21_regular_season_shots_made_per_zone_range.png)

Arremessos acertados por regiões da quadra (simplificado).

O atributo 'SHOT_ZONE_BASIC' oferece regiões da quadra, diferentes das presentes em 'SHOT_ZONE_AREA', utilizadas nessa plotagem.
    
![2020-21_regular_season_shots_made_per_zone_area_basic](https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/Images/2020-21_regular_season_shots_made_per_zone_area_basic.png)

# 4. Distribuição de arrmessos por distância e tipo de Arremesso (2 ou 3 pontos)

O gráfico abaixo apresenta a distribuição dos arremessos das 6 temporadas em análise, pela distância em que os arremessos foram feitos e pelo tipo de arremesso (2 ou 3 pontos).

Nele é fácil de se notar que a maioria das tentatidas de pontuação ocorre por arremessos de longa distância (atrás da linha de 3 pontos) ou por arremessos, bandejas ou enterradas feitos bem próximos da cesta. Os arremessos de média distância se tornaram algo do passado. 

![shot_distance_distribution](https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/Images/shot_distance_distribution.png)
    
    
# 5. Tipos de Arremessos
    
O gráfico a seguir mostra todos os arremessos tentados nas 6 temporadas pelo tipo de arremesso tentado (2 ou 3 pontos)

![shot_type_bar_plot.png](https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/Images/shot_type_bar_plot.png)


O gráfico a seguir mostra todos os arremessos tentados nas 6 temporadas pelo tipo de arremesso tentado (2 ou 3 pontos) e resultado do arremesso.

![shot_type_made_missed_bar_plot](https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/Images/shot_type_made_missed_bar_plot.png)


# 6. Arremessos por Tipo de Acão ofensiva

O gráfico abaixo mostra todos os arremesoss tentados nas 6 temporadas pelo tipo de ação ofensiva.

Nele nota-se que o 'jump shot' (ou arremesso) é o tipo de arremesso mais tentado na liga.

Para esclarecer a confusão que a tradução dos termos pode deixar, arremesso pode significar 'shot' que é um arremesso  qualquer ou 'jump shot' que é um tipo específico de arremesso. 

![shot_action_bar_plot](https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/Images/shot_action_bar_plot.png)
    
    
# Previsão de Arremessos utilizando modelos de Machine Learming

A partir destre ponto o conjunto de dados foi tratado a fim de alimentar modelos de Machine Learning com o intuito de prever o resultado de arremessos.


# Sub Conjuntos de Dados

Dada a dimensão do nosso conjunto principal de dados que possui, 989617 registros e 25 atributos. Foi decidido trabalhar com os modelos de machine learning utilizando sub conjuntos de dados.

Foram criadas duas funções 'choose_player' e 'choose_season'. 'choose_player' permite criar um sub conjunto de dados de um jogador da NBA. 'choose_season' permite criar um sub conjunto de dados de uma temporada da NBA.

Em ambas os sub conjuntos sofrem as seguintes operações:
    
* Redução de Dimensão: Os atributos 'PLAYER_NAME', 'EVENT_TYPE' e 'TEAM_NAME' são retirados por serem redundantes.

* Dummy Coding: Aplicado nos seguintes atrobutos categóricos  'GRID_TYPE', 'ACTION_TYPE', 'SHOT_TYPE', 'SHOT_ZONE_BASIC', 
      'SHOT_ZONE_AREA', 'SHOT_ZONE_RANGE', 'HTM', 'VTM' e 'SEASON_ID'.

* Train/Test split: Os conjuntos de Treino e Teste foram criados.

* Checagem e tratamento de atributos vom variância igual a zero: Foram calculadas as variâncias de todos os atributos e os com valor 0 foram retirados das bases de Treino e Teste.

* Normalização: Os dados das bases de Treino e Teste foram normalizados.


    

