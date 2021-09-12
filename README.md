# Análise Exploratória e Previsão de Arremessos da NBA

# Objetivo


* Este projeto visa primeiramente, realizar uma análise exploratória dos dados obtidos das últimas 6 temporadas regulares da NBA (2015-16 a 2020-21) e treinar diferentes modelos de machine learning com o intuito de prever se um arremesso é bem-sucedido ou não.
* Os dados foram obtidos através da API da NBA, o script 'get_players_shot_charts.ipynb' criado e a planilha com os ID's dos jogadores pode ser encontrados em: 

    - (https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA) 

* O notebook, chamado "NBA_SHOT_CHARTS", contendo todos os códigos desenvolvidos neste trabalho pode ser encontrado em:

   -    (https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/NBA_SHOT_CHARTS.ipynb)

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

As 10 planilhas foram importandas e inseridas em Dataframes utilizando a biblioteca pandas.

    #Read NBA Shots excel files

    nba_shots_2020_21 = pd.read_excel('nba_shots_2020-21.xlsx', engine='openpyxl')
    nba_shots_2019_20 = pd.read_excel('nba_shots_2019-20.xlsx', engine='openpyxl')
    nba_shots_2018_19 = pd.read_excel('nba_shots_2018-19.xlsx', engine='openpyxl')
    nba_shots_2017_18 = pd.read_excel('nba_shots_2017-18.xlsx', engine='openpyxl')
    nba_shots_2016_17 = pd.read_excel('nba_shots_2016-17.xlsx', engine='openpyxl')
    nba_shots_2015_16 = pd.read_excel('nba_shots_2015-16.xlsx', engine='openpyxl')
    nba_shots_2014_15 = pd.read_excel('nba_shots_2015-16.xlsx', engine='openpyxl')
    nba_shots_2013_14 = pd.read_excel('nba_shots_2015-16.xlsx', engine='openpyxl')
    nba_shots_2012_13 = pd.read_excel('nba_shots_2012-13.xlsx', engine='openpyxl')
    nba_shots_2011_12 = pd.read_excel('nba_shots_2011-12.xlsx', engine='openpyxl')
    
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
    
    #Drop "Unnamed: 0" column, Add "SEASON_ID" column in nba_shots_2014_15

    nba_shots_2014_15.drop(['Unnamed: 0'], axis=1, inplace=True)
    nba_shots_2014_15['SEASON_ID'] = '2014-15'
    nba_shots_2014_15.head()

    #Drop "Unnamed: 0" column, Add "SEASON_ID" column in nba_shots_2013_14

    nba_shots_2013_14.drop(['Unnamed: 0'], axis=1, inplace=True)
    nba_shots_2013_14['SEASON_ID'] = '2013-14'
    nba_shots_2013_14.head()

    #Drop "Unnamed: 0" column, Add "SEASON_ID" column in nba_shots_2012_13

    nba_shots_2012_13.drop(['Unnamed: 0'], axis=1, inplace=True)
    nba_shots_2012_13['SEASON_ID'] = '2012-13'
    nba_shots_2012_13.head()

    #Drop "Unnamed: 0" column, Add "SEASON_ID" column in nba_shots_2011_12

    nba_shots_2011_12.drop(['Unnamed: 0'], axis=1, inplace=True)
    nba_shots_2011_12['SEASON_ID'] = '2011-12'
    nba_shots_2011_12.head()

Os 10 Dataframes são concatenados em um único novo Dataframe chamado 'nba_shots'.

    #Create nba_shots as a concatenation of the 10 Dataframes from each reagular season

    nba_shots = pd.concat([nba_shots_2020_21, nba_shots_2019_20, nba_shots_2018_19, nba_shots_2017_18, nba_shots_2016_17, nba_shots_2015_16, nba_shots_2014_15, nba_shots_2013_14,
                            nba_shots_2012_13, nba_shots_2011_12], sort=False)

    nba_shots.head()


# Análise Inicial de nba_shots

    O dataset nba_shots possui 1345097 registros e 25 atributos.

    #Get nba_shots dataframe shape

    nba_shots.shape
    
    (1345097, 25)
    
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

    # Function to draw basketball court

    def create_court(ax, color):

        # Short corner 3PT lines
        ax.plot([-220, -220], [0, 140], linewidth=2, color=color)
        ax.plot([220, 220], [0, 140], linewidth=2, color=color)

        # 3PT Arc
        ax.add_artist(mpl.patches.Arc((0, 140), 440, 315, theta1=0, theta2=180, facecolor='none', edgecolor=color, lw=2))

        # Lane and Key
        ax.plot([-80, -80], [0, 190], linewidth=2, color=color)
        ax.plot([80, 80], [0, 190], linewidth=2, color=color)
        ax.plot([-60, -60], [0, 190], linewidth=2, color=color)
        ax.plot([60, 60], [0, 190], linewidth=2, color=color)
        ax.plot([-80, 80], [190, 190], linewidth=2, color=color)
        ax.add_artist(mpl.patches.Circle((0, 190), 60, facecolor='none', edgecolor=color, lw=2))

        # Rim
        ax.add_artist(mpl.patches.Circle((0, 60), 15, facecolor='none', edgecolor=color, lw=2))

        # Backboard
        ax.plot([-30, 30], [40, 40], linewidth=2, color=color)

        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Set axis limits
        ax.set_xlim(-250, 250)
        ax.set_ylim(0, 470)

        # General plot parameters
        #mpl.rcParams['font.family'] = 'Avenir'
        mpl.rcParams['font.size'] = 18
        mpl.rcParams['axes.linewidth'] = 2

        return ax
    

# Análise Exploratória


# 1. Arremessos por Jogador

De ínicio foram plotados todos os arremessos tentados na temporada 2020-21 de 3 atletas da liga (James Harden, Stephen Curry e Nikola Jokic). 

    # JAMES HARDEN 2020-21 REGULAR SEASON SHOTS

        # Create figure and axes
        fig = plt.figure(figsize=(10, 9))
        ax = fig.add_axes([0, 0, 1, 1])

        # Draw court
        ax = create_court(ax, 'black')

        # Shots Scatter Plots
        ax.scatter(nba_shots[(nba_shots['SEASON_ID'] == '2020-21') & (nba_shots['SHOT_MADE_FLAG']==1) & (nba_shots['PLAYER_NAME'] == 'James Harden')]['LOC_X'],
                    nba_shots[(nba_shots['SEASON_ID'] == '2020-21') & (nba_shots['SHOT_MADE_FLAG']==1) & (nba_shots['PLAYER_NAME'] == 'James Harden')]['LOC_Y'] +60, marker = "o", color = "Green")

        ax.scatter(nba_shots[(nba_shots['SEASON_ID'] == '2020-21') & (nba_shots['SHOT_MADE_FLAG']==0) & (nba_shots['PLAYER_NAME'] == 'James Harden')]['LOC_X'],
                    nba_shots[(nba_shots['SEASON_ID'] == '2020-21') & (nba_shots['SHOT_MADE_FLAG']==0) & (nba_shots['PLAYER_NAME'] == 'James Harden')]['LOC_Y'] +60, marker = "x", color = "Red")

        plt.title('JAMES HARDEN 2020-21 REGULAR SEASON SHOTS', fontsize = 20)
        plt.show()
 
![james_harden_2020-21_shot_chart](https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/Images/james_harden_2020-21_shot_chart.png)

     # SEPHEN CURRY 2020-21 REGULAR SEASON SHOTS

        # Create figure and axes
        fig = plt.figure(figsize=(10, 9))
        ax = fig.add_axes([0, 0, 1, 1])

        # Draw court
        ax = create_court(ax, 'black')

        # Plot scatter of shots
        ax.scatter(nba_shots[(nba_shots['SEASON_ID'] == '2020-21') & (nba_shots['SHOT_MADE_FLAG']==1) & (nba_shots['PLAYER_NAME'] == 'Stephen Curry')]['LOC_X'],
                    nba_shots[(nba_shots['SEASON_ID'] == '2020-21') & (nba_shots['SHOT_MADE_FLAG']==1) & (nba_shots['PLAYER_NAME'] == 'Stephen Curry')]['LOC_Y'] +60, marker = "o", color = "Green")

        ax.scatter(nba_shots[(nba_shots['SEASON_ID'] == '2020-21') & (nba_shots['SHOT_MADE_FLAG']==0) & (nba_shots['PLAYER_NAME'] == 'Stephen Curry')]['LOC_X'],
                    nba_shots[(nba_shots['SEASON_ID'] == '2020-21') & (nba_shots['SHOT_MADE_FLAG']==0) & (nba_shots['PLAYER_NAME'] == 'Stephen Curry')]['LOC_Y'] +60, marker = "x", color = "Red")

        plt.title('STEPHEN CURRY 2020-21 REGULAR SEASON SHOTS', fontsize = 20)
        plt.show()

![stephen_curry_2020-21_shot_chart](https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/Images/stephen_curry_2020-21_shot_chart.png)

       # NIKOLA JOKIC 2020-21 REGULAR SEASON SHOTS

        # Create figure and axes
        fig = plt.figure(figsize=(10, 9))
        ax = fig.add_axes([0, 0, 1, 1])

        # Draw court
        ax = create_court(ax, 'black')

        # Plot scatter of shots
        ax.scatter(nba_shots[(nba_shots['SEASON_ID'] == '2020-21') & (nba_shots['SHOT_MADE_FLAG']==1) & (nba_shots['PLAYER_NAME'] == 'Nikola Jokic')]['LOC_X'],
                    nba_shots[(nba_shots['SEASON_ID'] == '2020-21') & (nba_shots['SHOT_MADE_FLAG']==1) & (nba_shots['PLAYER_NAME'] == 'Nikola Jokic')]['LOC_Y'] +60, marker = "o", color = "Green")

        ax.scatter(nba_shots[(nba_shots['SEASON_ID'] == '2020-21') & (nba_shots['SHOT_MADE_FLAG']==0) & (nba_shots['PLAYER_NAME'] == 'Nikola Jokic')]['LOC_X'],
                    nba_shots[(nba_shots['SEASON_ID'] == '2020-21') & (nba_shots['SHOT_MADE_FLAG']==0) & (nba_shots['PLAYER_NAME'] == 'Nikola Jokic')]['LOC_Y'] +60, marker = "x", color = "Red")

        plt.title('NIKOLA JOKIC 2020-21 REGULAR SEASON SHOTS', fontsize = 20)
        plt.show()

![nikola_jokic_2020-21_shot_chart](https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/Images/nikola_jokic_2020-21_shot_chart.png)

  
# 2. Arremessos Acertados por Temporada

Abaixo temos as plotagens utilizando a função .hexbin da biblioteca matplotlib.

Nessa sequência de gráficos possível notar como o arremesso de 3 pontos se tornou cada vez mais o arremesso* mais popular na liga com o passar das temporadas.

*Arremessos não incluem ações ofensivas como bandejas e enterradas, que são feitas próximas da cesta e que continuam proeminentes na liga como pode ser notado em todas as imagens.

    # 2011-12 REGULAR SEASON MADE SHOTS

    # Create figure and axes
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_axes([0, 0, 1, 1])

    # Draw court
    ax = create_court(ax, 'black')

    # Plot scatter of shots
    ax.hexbin(nba_shots[(nba_shots['SEASON_ID'] == '2011-12') & (nba_shots['SHOT_MADE_FLAG']==1)]['LOC_X'],
                nba_shots[(nba_shots['SEASON_ID'] == '2011-12') & (nba_shots['SHOT_MADE_FLAG']==1)]['LOC_Y'] +60, gridsize=(30, 30), extent=(-300, 300, 0, 940), bins='log', cmap='Greens')

    plt.title('2011-12 REGULAR SEASON MADE SHOTS', fontsize = 20)
    plt.show()

![2011-12_regular_season_made_shots](https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/Images/2011-12_regular_season_made_shots.png)

    # 2012-13 REGULAR SEASON MADE SHOTS

    # Create figure and axes
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_axes([0, 0, 1, 1])

    # Draw court
    ax = create_court(ax, 'black')

    # Plot scatter of shots
    ax.hexbin(nba_shots[(nba_shots['SEASON_ID'] == '2012-13') & (nba_shots['SHOT_MADE_FLAG']==1)]['LOC_X'],
                nba_shots[(nba_shots['SEASON_ID'] == '2012-13') & (nba_shots['SHOT_MADE_FLAG']==1)]['LOC_Y'] +60, gridsize=(30, 30), extent=(-300, 300, 0, 940), bins='log', cmap='Greens')

    plt.title('2012-13 REGULAR SEASON MADE SHOTS', fontsize = 20)
    plt.show() 

![2012-13_regular_season_made_shots](https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/Images/2012-13_regular_season_made_shots.png)

    # 2013-14 REGULAR SEASON MADE SHOTS

    # Create figure and axes
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_axes([0, 0, 1, 1])

    # Draw court
    ax = create_court(ax, 'black')

    # Plot scatter of shots
    ax.hexbin(nba_shots[(nba_shots['SEASON_ID'] == '2013-14') & (nba_shots['SHOT_MADE_FLAG']==1)]['LOC_X'],
                nba_shots[(nba_shots['SEASON_ID'] == '2013-14') & (nba_shots['SHOT_MADE_FLAG']==1)]['LOC_Y'] +60, gridsize=(30, 30), extent=(-300, 300, 0, 940), bins='log', cmap='Greens')

    plt.title('2013-14 REGULAR SEASON MADE SHOTS', fontsize = 20)
    plt.show()

![2013-14_regular_season_made_shots](https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/Images/2013-14_regular_season_made_shots.png)

    # 2014-15 REGULAR SEASON MADE SHOTS

    # Create figure and axes
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_axes([0, 0, 1, 1])

    # Draw court
    ax = create_court(ax, 'black')

    # Plot scatter of shots
    ax.hexbin(nba_shots[(nba_shots['SEASON_ID'] == '2014-15') & (nba_shots['SHOT_MADE_FLAG']==1)]['LOC_X'],
                nba_shots[(nba_shots['SEASON_ID'] == '2014-15') & (nba_shots['SHOT_MADE_FLAG']==1)]['LOC_Y'] +60, gridsize=(30, 30), extent=(-300, 300, 0, 940), bins='log', cmap='Greens')

    plt.title('2014-15 REGULAR SEASON MADE SHOTS', fontsize = 20)
    plt.show()

![2014-15_regular_season_made_shots](https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/Images/2014-15_regular_season_made_shots.png)

    # 2015-16 REGULAR SEASON MADE SHOTS

    # Create figure and axes
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_axes([0, 0, 1, 1])

    # Draw court
    ax = create_court(ax, 'black')

    # Plot scatter of shots
    ax.hexbin(nba_shots[(nba_shots['SEASON_ID'] == '2015-16') & (nba_shots['SHOT_MADE_FLAG']==1)]['LOC_X'],
                nba_shots[(nba_shots['SEASON_ID'] == '2015-16') & (nba_shots['SHOT_MADE_FLAG']==1)]['LOC_Y'] +60, gridsize=(30, 30), extent=(-300, 300, 0, 940), bins='log', cmap='Greens')

    plt.title('2015-16 REGULAR SEASON MADE SHOTS', fontsize = 20)
    plt.show()

![2015-16_regular_season_made_shots](https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/Images/2015-16_regular_season_made_shots.png)

    # 2016-17 REGULAR SEASON MADE SHOTS

    # Create figure and axes
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_axes([0, 0, 1, 1])

    # Draw court
    ax = create_court(ax, 'black')

    # Plot scatter of shots
    ax.hexbin(nba_shots[(nba_shots['SEASON_ID'] == '2016-17') & (nba_shots['SHOT_MADE_FLAG']==1)]['LOC_X'],
                nba_shots[(nba_shots['SEASON_ID'] == '2016-17') & (nba_shots['SHOT_MADE_FLAG']==1)]['LOC_Y'] +60, gridsize=(30, 30), extent=(-300, 300, 0, 940), bins='log', cmap='Greens')

    plt.title('2016-17 REGULAR SEASON MADE SHOTS', fontsize = 20)
    plt.show()

![2016-17_regular_season_made_shots](https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/Images/2016-17_regular_season_made_shots.png)

    # 2017-18 REGULAR SEASON MADE SHOTS

    # Create figure and axes
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_axes([0, 0, 1, 1])

    # Draw court
    ax = create_court(ax, 'black')

    # Plot scatter of shots
    ax.hexbin(nba_shots[(nba_shots['SEASON_ID'] == '2017-18') & (nba_shots['SHOT_MADE_FLAG']==1)]['LOC_X'],
                nba_shots[(nba_shots['SEASON_ID'] == '2017-18') & (nba_shots['SHOT_MADE_FLAG']==1)]['LOC_Y'] +60, gridsize=(30, 30), extent=(-300, 300, 0, 940), bins='log', cmap='Greens')

    plt.title('2017-18 REGULAR SEASON MADE SHOTS', fontsize = 20)
    plt.show()

![2017-18_regular_season_made_shots](https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/Images/2017-18_regular_season_made_shots.png)

    # 2018-19 REGULAR SEASON MADE SHOTS

    # Create figure and axes
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_axes([0, 0, 1, 1])

    # Draw court
    ax = create_court(ax, 'black')

    # Plot scatter of shots
    ax.hexbin(nba_shots[(nba_shots['SEASON_ID'] == '2018-19') & (nba_shots['SHOT_MADE_FLAG']==1)]['LOC_X'],
                nba_shots[(nba_shots['SEASON_ID'] == '2018-19') & (nba_shots['SHOT_MADE_FLAG']==1)]['LOC_Y'] +60, gridsize=(30, 30), extent=(-300, 300, 0, 940), bins='log', cmap='Greens')

    plt.title('2018-19 REGULAR SEASON MADE SHOTS', fontsize = 20)
    plt.show()

![2018-19_regular_season_made_shots](https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/Images/2018-19_regular_season_made_shots.png)

    # 2019-20 REGULAR SEASON MADE SHOTS

    # Create figure and axes
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_axes([0, 0, 1, 1])

    # Draw court
    ax = create_court(ax, 'black')

    # Plot scatter of shots
    ax.hexbin(nba_shots[(nba_shots['SEASON_ID'] == '2019-20') & (nba_shots['SHOT_MADE_FLAG']==1)]['LOC_X'],
                nba_shots[(nba_shots['SEASON_ID'] == '2019-20') & (nba_shots['SHOT_MADE_FLAG']==1)]['LOC_Y'] +60, gridsize=(30, 30), extent=(-300, 300, 0, 940), bins='log', cmap='Greens')

    plt.title('2019-20 REGULAR SEASON MADE SHOTS', fontsize = 20)
    plt.show()

![2019-20_regular_season_made_shots](https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/Images/2019-20_regular_season_made_shots.png)

    # 2020-21 REGULAR SEASON MADE SHOTS

    # Create figure and axes
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_axes([0, 0, 1, 1])

    # Draw court
    ax = create_court(ax, 'black')

    # Plot scatter of shots
    ax.hexbin(nba_shots[(nba_shots['SEASON_ID'] == '2020-21') & (nba_shots['SHOT_MADE_FLAG']==1)]['LOC_X'],
                nba_shots[(nba_shots['SEASON_ID'] == '2020-21') & (nba_shots['SHOT_MADE_FLAG']==1)]['LOC_Y'] +60, gridsize=(30, 30), extent=(-300, 300, 0, 940), bins='log', cmap='Greens')

    plt.title('2020-21 REGULAR SEASON MADE SHOTS', fontsize = 20)
    plt.show()

![2020-21_regular_season_made_shots](https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/Images/2020-21_regular_season_made_shots.png)
    

# 3. Outras Visualizações de Arremessos

Os dados obtidos permitem ainda outras plotagens dos arremessos a seguir são mostradas 3 diferentes formas de enxergar os arremessos de acordo com sua posição em quadra.
    
Arremessos acertados por região da quadra

O atributo 'SHOT_ZONE_AREA' oferece as regiões da quadra utilizadas nessa plotagem.

    # 2020-21 REGULAR SEASON SHOTS MADE PER ZONE AREA

    # Create figure and axes
    fig = plt.figure(figsize=(20, 9))
    ax = fig.add_axes([0, 0, 1, 1])

    # Draw court
    ax = create_court(ax, 'black')

    # Plot scatter of shots
    sns.scatterplot(nba_shots[(nba_shots['SEASON_ID'] == '2020-21') & (nba_shots['SHOT_MADE_FLAG']==1)]['LOC_X'],
                nba_shots[(nba_shots['SEASON_ID'] == '2020-21') & (nba_shots['SHOT_MADE_FLAG']==1)]['LOC_Y'] + 60, hue = nba_shots[(nba_shots['SEASON_ID'] == '2020-21') & (nba_shots['SHOT_MADE_FLAG']==1)]['SHOT_ZONE_AREA'])

    plt.title('2020-21 REGULAR SEASON SHOTS MADE PER ZONE AREA', fontsize = 20)
    plt.show()

![2020-21_regular_season_shots_made_per_zone_area](https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/Images/2020-21_regular_season_shots_made_per_zone_area.png)
    
Arremessos acertados por zonas de distância.

O atributo 'SHOT_ZONE_RANGE' oferece as zonas por diferentes distâncias utilizadas nessa plotagem.

    # 2020-21 REGULAR SEASON SHOTS MADE PER ZONE RANGE

    # Create figure and axes
    fig = plt.figure(figsize=(20, 9))
    ax = fig.add_axes([0, 0, 1, 1])

    # Draw court
    ax = create_court(ax, 'black')

    # Plot scatter of shots
    sns.scatterplot(nba_shots[(nba_shots['SEASON_ID'] == '2020-21') & (nba_shots['SHOT_MADE_FLAG']==1)]['LOC_X'],
                nba_shots[(nba_shots['SEASON_ID'] == '2020-21') & (nba_shots['SHOT_MADE_FLAG']==1)]['LOC_Y'] + 60, hue = nba_shots[(nba_shots['SEASON_ID'] == '2020-21') & (nba_shots['SHOT_MADE_FLAG']==1)]['SHOT_ZONE_RANGE'])

    plt.title('2020-21 REGULAR SEASON SHOTS MADE PER ZONE RANGE', fontsize = 20)
    plt.show()

![2020-21_regular_season_shots_made_per_zone_range](https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/Images/2020-21_regular_season_shots_made_per_zone_range.png)

Arremessos acertados por regiões da quadra (simplificado).

O atributo 'SHOT_ZONE_BASIC' oferece regiões da quadra, diferentes das presentes em 'SHOT_ZONE_AREA', utilizadas nessa plotagem.

    # 2020-21 REGULAR SEASON SHOTS MADE PER ZONE AREA (BASIC)

    # Create figure and axes
    fig = plt.figure(figsize=(20, 9))
    ax = fig.add_axes([0, 0, 1, 1])

    # Draw court
    ax = create_court(ax, 'black')

    # Plot scatter of shots
    sns.scatterplot(nba_shots[(nba_shots['SEASON_ID'] == '2020-21') & (nba_shots['SHOT_MADE_FLAG']==1)]['LOC_X'],
                nba_shots[(nba_shots['SEASON_ID'] == '2020-21') & (nba_shots['SHOT_MADE_FLAG']==1)]['LOC_Y'] + 60, hue = nba_shots[(nba_shots['SEASON_ID'] == '2020-21') & (nba_shots['SHOT_MADE_FLAG']==1)]['SHOT_ZONE_BASIC'])

    plt.title('2020-21 REGULAR SEASON SHOTS MADE PER ZONE AREA (BASIC)', fontsize = 20)
    plt.show()
    
![2020-21_regular_season_shots_made_per_zone_area_basic](https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/Images/2020-21_regular_season_shots_made_per_zone_area_basic.png)

Os três gráficos acima nos mostram como os arremessos de 3 pontos se tornaram a principal forma de ate

# 4. Distribuição de arrmessos por distância e tipo de Arremesso (2 ou 3 pontos)

O gráfico abaixo apresenta a distribuição dos arremessos das 10 temporadas em análise, pela distância em que os arremessos foram feitos e pelo tipo de arremesso (2 ou 3 pontos).

Nele é fácil de se notar que a maioria das tentatidas de pontuação ocorre por arremessos de longa distância (atrás da linha de 3 pontos) ou por arremessos, bandejas ou enterradas feitos bem próximos da cesta. Os arremessos de média distância se tornaram bem menos utilizados.

    # SHOT DISTANCE DISTRIBUTION PLOT

    plt.figure(figsize=(20,12))
    fig1 = sns.histplot(data=nba_shots, x='SHOT_DISTANCE', hue = 'SHOT_TYPE')
    fig1.set_xlabel('SHOT_DISTANCE', fontsize=20)
    fig1.set_ylabel('COUNT', fontsize=20)
    fig1.tick_params(labelsize=15)
    plt.title('SHOT DISTANCE DISTRIBUTION', fontsize = 20)
    plt.xlim(0,40)
    plt.show()

![shot_distance_distribution](https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/Images/shot_distance_distribution.png)

Analisando as plotagens abaixo, distribuições tais como a anterior só que agora específicas para as temporadas 2011-21 e 2020-21, respectivamente primeira e última temporadas do nosso conjunto de dados, notamos com clareza a mudança no padrão das ações ofensivas com o passar dos anos. Arremessos de média distância deram espaço para os arremesos de 3 pontos.

    # SHOT DISTANCE DISTRIBUTION PLOT 2011-12 SEASON

    plt.figure(figsize=(20,12))
    fig2 = sns.histplot(data=nba_shots, 
                        x=nba_shots[nba_shots['SEASON_ID'] == '2011-12']['SHOT_DISTANCE'], 
                        hue = nba_shots[nba_shots['SEASON_ID'] == '2011-12']['SHOT_TYPE'])
    fig2.set_xlabel('SHOT_DISTANCE', fontsize=20)
    fig2.set_ylabel('COUNT', fontsize=20)
    fig2.tick_params(labelsize=15)
    plt.title('SHOT DISTANCE DISTRIBUTION 2011-12 SEASON', fontsize = 20)
    plt.xlim(0,40)
    plt.show()

![shot_distance_distribution_2011-12](https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/Images/shot_distance_distribution_2011-12.png)

    # SHOT DISTANCE DISTRIBUTION PLOT 2020-21 SEASON

    plt.figure(figsize=(20,12))
    fig3 = sns.histplot(data=nba_shots, 
                        x=nba_shots[nba_shots['SEASON_ID'] == '2020-21']['SHOT_DISTANCE'], 
                        hue = nba_shots[nba_shots['SEASON_ID'] == '2020-21']['SHOT_TYPE'])
    fig3.set_xlabel('SHOT_DISTANCE', fontsize=20)
    fig3.set_ylabel('COUNT', fontsize=20)
    fig3.tick_params(labelsize=15)
    plt.title('SHOT DISTANCE DISTRIBUTION 2020-21 SEASON', fontsize = 20)
    plt.xlim(0,40)
    plt.show()

![shot_distance_distribution_2020-21](https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/Images/shot_distance_distribution_2020-21.png)
    
    
# 5. Tipos de Arremessos
    
O gráfico a seguir mostra todos os arremessos tentados nas 10 temporadas pelo tipo de arremesso tentado (2 ou 3 pontos)

    # SHOT TYPE BAR PLOT

    plt.figure(figsize=(20,12))
    fig4 = sns.countplot(data=nba_shots, x='SHOT_TYPE', palette = 'husl')
    fig4.set_xlabel('SHOT_TYPE', fontsize=20)
    fig4.set_ylabel('COUNT', fontsize=20)
    fig4.tick_params(labelsize=20)
    plt.title('SHOT TYPE', fontsize = 20)
    for p in fig4.patches:
        txt = str(p.get_height().round(2))
        txt_x = p.get_x() 
        txt_y = p.get_height()
        fig4.text(txt_x,txt_y,txt)
    plt.show()

![shot_type_bar_plot.png](https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/Images/shot_type_bar_plot.png)

O gráfico a seguir mostra todos os arremessos tentados nas 10 temporadas pelo tipo de arremesso tentado (2 ou 3 pontos) e resultado do arremesso.

    # SHOT TYPE (MADE/MISSED) BAR PLOT

    plt.figure(figsize=(20,12))
    fig5 = sns.countplot(data=nba_shots, x='SHOT_TYPE', palette = 'husl', hue = 'EVENT_TYPE')
    fig5.set_xlabel('SHOT_TYPE', fontsize=20)
    fig5.set_ylabel('COUNT', fontsize=20)
    fig5.tick_params(labelsize=20)
    plt.title('SHOT TYPE (MADE/MISSED)', fontsize = 20)
    for p in fig5.patches:
        txt = str(p.get_height().round(2))
        txt_x = p.get_x() 
        txt_y = p.get_height()
        fig5.text(txt_x,txt_y,txt)
    plt.show()

![shot_type_made_missed_bar_plot](https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/Images/shot_type_made_missed_bar_plot.png)

Análogo ao que foi feito com os gráficos de distribuição, como pode ser visto abaixo, plotando os gráficos de barra por tipo de arremesso vemos que proporcionalmente a quantidade de arremessos de 2 e 3 pontos é muito mais próxima na temporada 2020-21 do que era na temporada 2011-12.

    # SHOT TYPE BAR PLOT

    plt.figure(figsize=(20,12))
    fig6 = sns.countplot(data=nba_shots, x=nba_shots[nba_shots['SEASON_ID'] == '2011-12']['SHOT_TYPE'], palette = 'husl')
    fig6.set_xlabel('SHOT_TYPE', fontsize=20)
    fig6.set_ylabel('COUNT', fontsize=20)
    fig6.tick_params(labelsize=20)
    plt.title('SHOT TYPE 2011-12 SEASON', fontsize = 20)
    for p in fig6.patches:
        txt = str(p.get_height().round(2))
        txt_x = p.get_x() 
        txt_y = p.get_height()
        fig6.text(txt_x,txt_y,txt)
    plt.show()
    
![shot_type_2011-12_season_bar_plot](https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/Images/shot_type_2011-12_season_bar_plot.png)

    # SHOT TYPE BAR PLOT

    plt.figure(figsize=(20,12))
    fig7 = sns.countplot(data=nba_shots, x=nba_shots[nba_shots['SEASON_ID'] == '2020-21']['SHOT_TYPE'], palette = 'husl')
    fig7.set_xlabel('SHOT_TYPE', fontsize=20)
    fig7.set_ylabel('COUNT', fontsize=20)
    fig7.tick_params(labelsize=20)
    plt.title('SHOT TYPE 2020-21 SEASON', fontsize = 20)
    for p in fig7.patches:
        txt = str(p.get_height().round(2))
        txt_x = p.get_x() 
        txt_y = p.get_height()
        fig7.text(txt_x,txt_y,txt)
    plt.show()
    
![shot_type_2020-21_season_bar_plot](https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/Images/shot_type_2020-21_season_bar_plot.png)

# 6. Arremessos por Tipo de Acão ofensiva

O gráfico abaixo mostra todos os arremesoss tentados nas 10 temporadas pelo tipo de ação ofensiva.

Nele nota-se que o 'jump shot' (ou arremesso) é o tipo de arremesso mais tentado na liga.

Para esclarecer a confusão que a tradução dos termos pode deixar, arremesso pode significar 'shot' que é um arremesso  qualquer ou 'jump shot' que é um tipo específico de arremesso. 

![shot_action_bar_plot](https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/Images/shot_action_bar_plot.png)

# 7. Arremessos por Período

Outra forma interessante de se enxergar a efetividade dos arremessos é através dos períodos de um jogo de basquete. Um jogo tem 4 períodos e se ao final o jogo permanecer empatado, períodos extra mais curtos, são jogados até que ao final de um deles um time esteja vencendo.

Abaixo podemos ver que média de acerto cai com o avanço dos períodos, algo que pode-se considerar esperado. Já que com o passar do jogo, cada arremesso tende a carregar maior importância, sendo esse um aspecto mental que pode afetar os atletas. Outro fator é o físico, quando mais se joga mais cansados estão os atletas, o que os leva em momentos a não conseguir performar o movimento do arremesso corretamente. 
    
    # SHOT PER GAME PERIOD BAR PLOT

    plt.figure(figsize=(20,12))
    fig5 = sns.countplot(data=nba_shots, x='PERIOD', palette = 'husl', hue = 'EVENT_TYPE')
    fig5.set_xlabel('SHOT_TYPE', fontsize=20)
    fig5.set_ylabel('COUNT', fontsize=20)
    fig5.tick_params(labelsize=20)
    plt.title('SHOT TYPE PER GAME PERIOD', fontsize = 20)
    for p in fig5.patches:
        txt = str(p.get_height().round(2))
        txt_x = p.get_x() 
        txt_y = p.get_height()
        fig5.text(txt_x,txt_y,txt)
    plt.show()
    
![shot_type_per_game_period](https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/Images/shot_type_per_game_period.png)

Média Primeiro Período: 169006/(191077 + 169006) = 0,4694 -> 46,94%

Média Segundo Período: 152177/(175035 + 152177) = 0,4650 -> 46,50%
 
Média Terceiro Período: 157762/(183722 + 157762) = 0,4620 -> 46,20%
 
Média Quarto Período: 138886/(167738 + 138886) = 0,4530 -> 45,30%
    
# Previsão de Arremessos utilizando modelos de Machine Learming

A partir destre ponto o conjunto de dados foi tratado a fim de alimentar modelos de Machine Learning com o intuito de prever o resultado de arremessos.


# Sub Conjuntos de Dados

Dada a dimensão do nosso conjunto principal de dados que possui, 1345097 registros e 25 atributos. Foi decidido trabalhar com os modelos de machine learning utilizando sub conjuntos de dados.

Foram criadas duas funções 'choose_player' e 'choose_season'. 'choose_player' permite criar um sub conjunto de dados de um jogador da NBA. 'choose_season' permite criar um sub conjunto de dados de uma temporada da NBA.

Em ambas funções, os sub conjuntos sofrem as seguintes operações:
    
* Redução de Dimensão: Os atributos 'PLAYER_NAME', 'EVENT_TYPE' e 'TEAM_NAME' são retirados por serem redundantes em informação fornecida com os atributos 'PLAYER_ID', 'SHOT_MADE_FLAG' e 'TEAM_ID' respectivamente.

* Dummy Coding: Aplicado nos seguintes atributos categóricos  'GRID_TYPE', 'ACTION_TYPE', 'SHOT_TYPE', 'SHOT_ZONE_BASIC', 
      'SHOT_ZONE_AREA', 'SHOT_ZONE_RANGE', 'HTM', 'VTM' e 'SEASON_ID'.

* Train/Test split: Os conjuntos de Treino e Teste foram criados com os seguintes parâmetros:

    - Razão treino/teste igual a 0.2
    - Random state igual a 100 para permitir reproducibilidade ods conjuntos de treino e teste.
    - Estratificação ativada para permitir uma divisão equilibrada entre os resultados do sub conjunto de dados escolhido.

* Checagem e tratamento de atributos vom variância igual a zero: Foram calculadas as variâncias de todos os atributos e os com valor 0 foram retirados das bases de Treino e Teste.

* Normalização: Os dados das bases de Treino e Teste foram normalizados. É uma etapa fundamental pois o conjunto de dados possui atributos numéricos em escalas bastante distintas, como por exmplo 'TEAM_ID' e 'PERIOD'. Essa diferença pode levar os modelos a uma menor eficiência.

Para o treinamento dos modelos que virão a seguir, iremos trabalhar com o sub conjunto de arremessos do jogador Stephen Curry. Treinamentos também foram feitos utilizando sub conjuntos de uma temporada inteira (2020-21) e os resultados não se distanciaram significativamente dos resultados obtidos utilizando apenas os arremessos do jogador.

# Análise exploratória do sub conjunto de dados

Decidido o sub conjunto de dados que iremos utilizar para o treinamento e avaliação dos nossos modelos, é interessante repetir agora algumas das avaliações que fizemos anteriormente, apenas considerando os arremessos do Stephen Curry.
    
Como pode ser visto na distribuição de arremessos por distância e tipo apresentada abaixo. O jogador possui um perfil de arremessar preferencialmente bolas de 3 pontos.

    # SHOT DISTANCE DISTRIBUTION PLOT STEPHEN CURRY

    plt.figure(figsize=(20,12))
    fig9 = sns.histplot(data=nba_shots, 
                        x=nba_shots[nba_shots['PLAYER_NAME'] == 'Stephen Curry']['SHOT_DISTANCE'], 
                        hue = nba_shots[nba_shots['PLAYER_NAME'] == 'Stephen Curry']['SHOT_TYPE'])
    fig9.set_xlabel('SHOT_DISTANCE', fontsize=20)
    fig9.set_ylabel('COUNT', fontsize=20)
    fig9.tick_params(labelsize=15)
    plt.title('SHOT DISTANCE DISTRIBUTION STEPHEN CURRY', fontsize = 20)
    plt.xlim(0,40)
    plt.show()
    
![shot_distance_distribution_stephen_curry](https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/Images/shot_distance_distribution_stephen_curry.png)

Nos gráficos de barra abaixo, que mostram os arremessos divididos pelo tipo (2 e 3 pontos) e por tipo e resultado respsctivamente. Notamos que o Stephen Curry foge do comportamento visto nestes mesmos gráficos que foram plotados considerando o dataset inteiro.

Em 10 temporadas, o atelta estreou na temporada 2009-10, Curry arremesou mais bolas de 3 pontos do que de 2. Algo impensável antes do surgimento do mesmo. Vale lembrar que ele já é considerado quase que unanimamente por mídia especializada e fãs, como o melhor arremessador de 3 pontos da história da liga. Tendo sob seu nome vários recordes como por exemplo, mais arremessos de 3 pontos acertados em uma temporada regular (2015-2016) e mais arremessos de 3 pontos acertados em playoffs (na história).[1]

    # SHOT TYPE BAR PLOT STEPHEN CURRY

    plt.figure(figsize=(20,12))
    fig10 = sns.countplot(data=nba_shots, x=nba_shots[nba_shots['PLAYER_NAME'] == 'Stephen Curry']['SHOT_TYPE'], palette = 'husl')
    fig10.set_xlabel('SHOT_TYPE', fontsize=20)
    fig10.set_ylabel('COUNT', fontsize=20)
    fig10.tick_params(labelsize=20)
    plt.title('SHOT TYPE STEPHEN CURRY', fontsize = 20)
    for p in fig10.patches:
        txt = str(p.get_height().round(2))
        txt_x = p.get_x() 
        txt_y = p.get_height()
        fig10.text(txt_x,txt_y,txt)
    plt.show()
    
![shot_type_bar_plot_stephen_curry](https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/Images/shot_type_bar_plot_stephen_curry.png) 


    # SHOT TYPE (MADE/MISSED) BAR PLOT STEPHEN CURRY

    plt.figure(figsize=(20,12))
    fig11 = sns.countplot(data=nba_shots, x=nba_shots[nba_shots['PLAYER_NAME'] == 'Stephen Curry']['SHOT_TYPE'], 
                        palette = 'husl', 
                        hue = nba_shots[nba_shots['PLAYER_NAME'] == 'Stephen Curry']['EVENT_TYPE'])
    fig11.set_xlabel('SHOT_TYPE', fontsize=20)
    fig11.set_ylabel('COUNT', fontsize=20)
    fig11.tick_params(labelsize=20)
    plt.title('SHOT TYPE (MADE/MISSED) STEPHEN CURRY', fontsize = 20)
    for p in fig11.patches:
        txt = str(p.get_height().round(2))
        txt_x = p.get_x() 
        txt_y = p.get_height()
        fig11.text(txt_x,txt_y,txt)
    plt.show()

![shot_type_made_missed_bar_plot_stephen_curry](https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/Images/shot_type_made_missed_bar_plot_stephen_curry.png) 

Apesar de sua extrema capacidade na bola de 3, Curry ainda sim acerta percentualmente mais bolas de 2 do que de 3, como pode ser visto no gráfico abaixo. 

Média nos arremessos de 2 pontos: 2840/(2840 + 2403) = 0,5417 -> 54,17%

Media nos arremessos de 3 pontos: 2768/(3554 + 2768) = 0,4378 -> 43,78%

Média geral: (2840 + 2768)/(2840 + 2403 + 3554 + 2768) = 0,4849 -> 48,49%

A diferença nos valores é absolutamente normal dada a natureza do jogo de basquete, a bola de 3 é uma ação ofensiva de maior risco e maior recompensa.

Hoje, os times notam que esse maior risco vale ser encarado. Tendo em consideração as médias do Curry. Como calculado, ele acerta em média 54,17% dos seus arremessos de 2 pontos, em um cenário em que tente 10 desses arremessos, ele acertaria 5 e isso resultaria num total de 10 pontos. Já se fizer as mesmas 10 tentativas para a bola de 3, com sua média de 43,78%, acertaria 4 bolas que somariam um total de 12 pontos.

Obviamente que esse cenário que ignora dezenas de fatores que levam ao sucesso ou não de um arremesso em um jogo oficial de basquete na NBA. Porém, essa lógica é a base do pensamento que levam hoje muitos dos times e sua comissões técnicas a priorizarem a bola de 3 mesmo que isso faça com que a média de acerto dos arremessos caia.

Diferente do que foi visto para o dataset geral, se analizarmos a performance de acerto de arremessos por período de jogo, Curry não aparenta seguir o mesmo padrão que o resto da liga, onde as médias de acerto caem com o avançar do jogo.

    # SHOT PER GAME PERIOD BAR PLOT

    plt.figure(figsize=(20,12))
    fig12 = sns.countplot(data=nba_shots, x=nba_shots[nba_shots['PLAYER_NAME'] == 'Stephen Curry']['PERIOD'],
                            palette = 'husl', 
                            hue = nba_shots[nba_shots['PLAYER_NAME'] == 'Stephen Curry']['EVENT_TYPE'])
    fig12.set_xlabel('SHOT_TYPE', fontsize=20)
    fig12.set_ylabel('COUNT', fontsize=20)
    fig12.tick_params(labelsize=20)
    plt.title('SHOT TYPE PER GAME PERIOD STEPHEN CURRY', fontsize = 20)
    for p in fig12.patches:
        txt = str(p.get_height().round(2))
        txt_x = p.get_x() 
        txt_y = p.get_height()
        fig12.text(txt_x,txt_y,txt)
    plt.show()
    
![shot_type_per_game_period_stephen_curry](https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/Images/shot_type_per_game_period_stephen_curry.png)

Funcão choose_player

        # Due to the large amount of the dataset, everything past this point you be done per player. The function below creates a sub dataset from nba_shots with the data form the chosen player. 

    def choose_player (player_name):
        player_shots = nba_shots[nba_shots['PLAYER_NAME'] == player_name]
        print(player_shots.head())

        # Dimensional Reduction: Columns PLAYER_ID and PLAYER_NAME carry the same type of information, PLAYER_NAME is going to be droped. 
        # The same happens to columns EVENT_TYPE and SHOT_MADE_FLAG, EVENT_TYPE is going to be droped.
        # It also happens for TEAM_ID and TEAM_NAME, TEAM_NAME is going to be droped.

        nba_shots_ml = player_shots.drop(['PLAYER_NAME', 'EVENT_TYPE', 'TEAM_NAME'], axis = 1)

        # Apply Dummy Coding to the categorial attributes of the dataset

        categorical_columns = ['GRID_TYPE', 'ACTION_TYPE', 'SHOT_TYPE', 'SHOT_ZONE_BASIC', 
        'SHOT_ZONE_AREA', 'SHOT_ZONE_RANGE', 'HTM', 'VTM', 'SEASON_ID']

        for i in categorical_columns:

            nba_shots_ml = pd.get_dummies(nba_shots_ml, columns=[i], drop_first=True)

        #Train/Test split

        X = nba_shots_ml.loc[:, nba_shots_ml.columns != 'SHOT_MADE_FLAG']
        y = nba_shots_ml['SHOT_MADE_FLAG']
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size= 0.3, 
                                                            random_state = 100, 
                                                            stratify = y,
                                                            )

        # Check columns with variance equal to zero and drop them

        zero_var_filter = VarianceThreshold()
        X_train = zero_var_filter.fit_transform(X_train)
        X_test = zero_var_filter.transform(X_test)
        print('X_train e X_test possuíam', (zero_var_filter.variances_ == 0).sum(), 'atributo(s) com variância igual a zero')

        print('X_train:', X_train.shape)
        print('X_test:', X_test.shape)
        print('y_train:', y_train.shape)
        print('y_test:', y_test.shape)

        # Normalize the data

        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    X_train, X_test, y_train, y_test = choose_player('Stephen Curry')
    
    
Função choose_season

        # Due to the large amount of the dataset, everything past this point you be done per season. The function below creates a sub dataset from nba_shots with the data form the chosen season. 

    def choose_season (season_id):
        season_shots = nba_shots[nba_shots['SEASON_ID'] == season_id]
        print(season_shots.head())

        # Dimensional Reduction: Columns PLAYER_ID and PLAYER_NAME carry the same type of information, PLAYER_NAME is going to be droped. 
        # The same happens to columns EVENT_TYPE and SHOT_MADE_FLAG, EVENT_TYPE is going to be droped.
        # It also happens for TEAM_ID and TEAM_NAME, TEAM_NAME is going to be droped.

        nba_shots_ml = season_shots.drop(['PLAYER_NAME', 'EVENT_TYPE', 'TEAM_NAME'], axis = 1)

        # Apply Dummy Coding to the categorial attributes of the dataset

        categorical_columns = ['GRID_TYPE', 'ACTION_TYPE', 'SHOT_TYPE', 'SHOT_ZONE_BASIC', 
        'SHOT_ZONE_AREA', 'SHOT_ZONE_RANGE', 'HTM', 'VTM', 'SEASON_ID']

        for i in categorical_columns:

            nba_shots_ml = pd.get_dummies(nba_shots_ml, columns=[i], drop_first=True)

        #Train/Test split

        X = nba_shots_ml.loc[:, nba_shots_ml.columns != 'SHOT_MADE_FLAG']
        y = nba_shots_ml['SHOT_MADE_FLAG']
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size= 0.2, 
                                                            random_state = 100, 
                                                            #stratify = y
                                                            )

        # Check columns with variance equal to zero and drop them

        zero_var_filter = VarianceThreshold()
        X_train = zero_var_filter.fit_transform(X_train)
        X_test = zero_var_filter.transform(X_test)
        print('X_train e X_test possuíam', (zero_var_filter.variances_ == 0).sum(), 'atributo(s) com variância igual a zero')

        print('X_train:', X_train.shape)
        print('X_test:', X_test.shape)
        print('y_train:', y_train.shape)
        print('y_test:', y_test.shape)

        # Normalize the data

        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    X_train, X_test, y_train, y_test = choose_season('2020-21')
    
    
Os conjuntos de Treino e Teste do sub conjunto de arremessos do Stephen Curry ficaram com os seguintes formatos:

* X_train e X_test possuíam 4 atributo(s) com variância igual a zero

Os conjuntos de Treino e Teste do sub conjunto de arremessos do Stephen Curry ficaram com os seguintes formatos:

* X_train: (9252, 138)
* X_test: (2313, 138)
* y_train: (9252,)
* y_test: (2313,)

O número de atributos cresceu consideravelmente dos 25 originais para os 138 do sub conjunto selecionado. O princpal fator para tamanha mudança foi a aaplicação do Dummy Coding que transformou os atributos categóricos em numéricos para poderem ser compreendidos pelos modelos.
    

Para todos os modelos treinados foi utilizado o GridSaerchCV do Sklearn para realizar a tunagem de Hiper-parâmetros. Foi realizada a busca pelos Hiper-parâmetros que resultassem na melhor acurácia.

Os códigos contendo o treinamentos, as predições e as avaliações dos modelos que virão a seguir estão presentes no notebook "NBA_SHOT_CHARTS" mencionado no início desse relatório. Aqui iremos apenas tratar dos resultados e dos parâmetros utilizados. 

Todos os modelos abaixo são provenientes da biblioteca SKLearn. A exceção fica para o modelo XGBOOST que pussui uma biblioteca própria de mesmo nome. 

# Modelos de Machine Learning

Todos os modelos de machine learning que serão usados nesse trabalho são do tipo supervisionado e de classificação. A escolha de quais tipos de modelos usar passa pela natureza dos dados que serão trabalhados como também o que deseja-se ser capaz de prever com tais modelos.

Nós estamos trabalhando com um dataset de arremessos de jogadores da NBA. O nosso conjunto possui o resultado de cada arremesso logo, faz sentido usarmos modelos supervisionados, aqueles que são treinados tendo conhecimento do resultado de cada registro a ser usado no treinamento. 

Dentro do grupo de modelos supervisionados, utilizaremos os de classificação já que, nosso intuito é poder prever se um arremesso entrou ou não.

# SVM

O modelo SVM/SVC (Support Vector Classification) conseguiu atingir 66% de acurácia e 63% de F1 score. Os melhores valores para os hiper-parâmetros utilizados foram: 

* random_state = 100
* C = 100 
* gamma = 0.0001
* kernel = 'rbf'

Abaixo, a matriz de confusão entre y_pred e y_test:

![confusion_matrix_SVM](https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/Images/confusion_matrix_SVM.png)
 

# Decision Tree

O modelo Decision Tree Classifier conseguiu atingir 66% de acurácia e 66% de F1 score. Os melhores valores para os hiper-parâmetros utilizados foram: 

* random_state = 100
* max_depth = 5  
* min_samples_leaf = 5
* min_samples_split = 10

Durante a etapa de tunagem de hiper-parâmetros, o modelo chegou a alcançar 72% de acurácia e F1 escore com max_depth = 20, min_samples_leaf = 1 e min_samples_split=2. Apesar do resultado superior, tais valores para estes hiper-parâmetros podem significar a criação de uma árvora de decisão muito especializada no conjunto de dados de treino.

Abaixo, a matriz de confusão entre y_pred e y_test:

![confusion_matrix_DT](https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/Images/confusion_matrix_DT.png)

Abaixo, pode-se ver a árvore de decisão do modelo treinado:

![DT_plot](https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/Images/DT_plot.png)

 
# Random Forest

O modelo Random Forest Classifier conseguiu atingir 70% de acurácia e 70% de F1 score. Os melhores valores para os hiper-parâmetros utilizados foram: 

* random_state = 100
* min_samples_leaf = 8
* min_samples_split = 10
* max_features = 'auto' 
* n_estimators = 100
* n_jobs = 50
* verbose = 1

Entretando, se adicionado o hiper-parâmetro max_depth (por padrão possui valor nulo), que foi usado no modelo de Árvore de Desisão, ocorre uma queda tanto da acurácia como do F1 score, já que este hiper-parâmetro define um limite para a profundidade das árvores a serem treinadas. Normalmente esti tipo de limitação é definida para evitar o overfitting do modelo. Por exemplo, para max_depth = 10, acurárica e F1 score caem por volta de 67% e para max_depth = 5, ambos ficam em torno de 65%.

Abaixo, a matriz de confusão entre y_pred e y_test:

![confusion_matrix_RF](https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/Images/confusion_matrix_RF.png)

Abaixo, pode-se ver a plotagem da importância dos atribudos do dataset usado no treino do modelo:

![Feature_Importances_RF](https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/Images/Feature_Importances_RF.png)


# XGBOOST

O modelo XGBOOST foi o que conseguiu melhores resultados, 72% de acurácia e 71% de F1 score. Os melhores valores para os hiper-parâmetros utilizados foram:

* random_state = 100
* n_estimators = 2000
* objective = "reg:logistic"
* use_label_encoder = False
* booster = 'gbtree'
* alpha = 10
* max_depth = 8
* gamma = 1e-2
* colsample_bytree = 0.5

Abaixo, a matriz de confusão entre y_pred e y_test:

![confusion_matrix_XGBOOST](https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA/blob/main/Images/confusion_matrix_XGBOOST.png)

# Conclusão

Após a avaliação de todos os modelos treinados, ficou claro que os modelos do tipo Comitê são os melhores entre os testados. Random Forest Classifier, que utiliza o método de bagging e XGBOOST, que utiliza boosting, foram os que obtiveram as melhores performances. Os resultados foram os esperados se considerarmos que os modelos do tipo Comitê são reconhecidamente bons para cenários onde temos um conjunto muito grande de dados e o problema é muito complexo. A complexidade do problema é um fator a ser bastante considerado pois, estamos tratando de de um tipo de evento, arremesso de basquete, que possui ínumeras variáveis que compõe e afetam o resultado.

Esses resultados corroboram o favoritismo desses tipos de modelos para a previsão de arremessos. Resultados semelhantes foram encontrados por Brett Meehan em "Predicting NBA Shots" e Max Murakami-Moses em "Analysis of Machine Learning Models Predicting
Basketball Shot Success". [2][3]

# Referências

    [1] Stephen Curry career stats. https://www.basketball-reference.com/players/c/curryst01.html, 2021.

    [2] B. Meehan. Predicting NBA Shots. http://cs229.stanford.edu/proj2017/final-reports/5132133.pdf.

    [3] M. Murakami-Moses. Analysis of Machine Learning Models Predicting Basketball Shot Success. https://www.the-iyrc.org/uploads/1/2/9/7/129787256/20_iyrc2020_35_final.pdf.



