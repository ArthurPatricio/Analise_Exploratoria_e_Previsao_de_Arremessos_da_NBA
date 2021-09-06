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
    
# Leitura dos Dados

As 6 planilhas foram importandas e inseridas em Dataframes utilizando a biblioteca pandas.
    
Cada Dataframe tem a coluna "Unnamed: 0" retirada e a coluna "SEASON_ID" adicionada sendo inserida a respectiva temporada do Dataframe em questão.

Os 6 Dataframes são concatenados em um único novo Dataframe chamado 'nba_shots'.


# Análise Inicial de nba_shots

# Checagem de valores nulos

nba_shots não possui nenhum valor faltante.

Foi utulizado a biblioteca missingno para realizar a checagem.
    
    
# Relatório Pandas Profile

Foi gerado o 'Pandas Profile Report' que oferece uma análise extensa do conjunto de dados que está sendo abordado.
    

# Função para desenhar a quadra

A função 'create_court' abaixo foi obtida do seguinte artigo: 
    
- (https://towardsdatascience.com/make-a-simple-nba-shot-chart-with-python-e5d70db45d0d)

Esta função cria desenha uma quadra de basquete nas proporções da NBA utilizando matplotlib.
    

# Análise Exploratória


# 1.Arremessos por Jogador

De ínicio foram plotados todos os arremessos tentados na temporada 20-21 de 3 atletas da liga (James Harden, Stephen Curry e Nikola Jokic). 
    

# 2.Arremessos Acertados por Temporada

Abaixo temos as plotagens utilizando a função .hexbin da biblioteca matplotlib.

Nessa sequência de gráficos possível notar como o arremesso de 3 pontos se tornou cada vez mais o arremesso* mais popular na liga.

*Arremessos não incluem ações ofensivas como bandejas e enterradas, que são feitas próximas da cesta e que continuam proeminentes na liga como pode ser notado em todas as imagens.
    

# 3.Outras Visualizações de Arremessos

Os dados obtidos permitem ainda outras plotagens dos arremessos a seguir são mostradas 3 diferentes formas de enxergar os arremessos de acordo com sua posição em quadra.
    

Arremessos acertados por região da quadra

O atributo 'SHOT_ZONE_AREA' oferece as regiões da quadra utilizadas nessa plotagem.
    
Arremessos acertados por zonas de distância.


O atributo 'SHOT_ZONE_RANGE' oferece as zonas por diferentes distâncias utilizadas nessa pltagem.
    
    
Arremessos acertados por regiões da quadra (simplificado).

O atributo 'SHOT_ZONE_BASIC' oferece regiões da quadra, diferentes das presentes em 'SHOT_ZONE_AREA', utilizadas nessa plotagem.
    

# 4.Distribuição de arrmessos por distância e tipo de Arremesso (2 ou 3 pontos)

O gráfico abaixo apresenta a distribuição dos arremessos das 6 temporadas em análise, pela distância em que os arremessos foram feitos e pelo tipo de arremesso ((2 ou 3 pontos)).

Nele é fácil de se notar que a maioria das tentatidas de pontuação ocorre por arremessos de longa distância (atrás da linha de 3 pontos) ou por arremessos, bandejas ou enterradas feitos bem próximos da cesta. Os arremessos de média distância se tornaram algo do passado. 
    
    
# 5. Tipos de Arremessos
    
O gráfico a seguir mostra todos os arremessos tentados nas 6 temporadas pelo tipo de arremesso tentado (2 ou 3 pontos)


O gráfico a seguir mostra todos os arremessos tentados nas 6 temporadas pelo tipo de arremesso tentado (2 ou 3 pontos) e resultado do arremesso.


# 5.Arremessos por Tipo de Acão ofensiva

O gráfico abaixo mostra todos os arremesoss tentados nas 6 temporadas pelo tipo de ação ofensiva.

Nele nota-se que o 'jump shot' (ou arremesso) é o tipo de arremesso mais tentado na liga.

Ps: Esclarecendo a confusão que a tradução dos termos pode deixar, arremesso pode significar 'shot' que é um arremesso  qualquer ou 'jump shot' que é um tipo específico de arremesso. 
    
    
# Previsão de Arremessos utulizando modelos de Machine Learming

A partir destre ponto o conjunto de dados foi tratado a fim de alimentar modelos de Machine Learning com o intuito de prever o resultado de arremessos.


# Sub Conjuntos de Dados

Dada a dimensão do nosso conjunto principal de dados que possui, 989617 registros e 25 atributos. Foi decidido trabalhar com os modelos de machine learning utilizando sub conjuntos de dados.

Foram criadas duas funções 'choose_player' e 'choose_season'. 'choose_player' permite criar um sub conjunto de dados de um jogador da NBA. 'choose_season' permite criar um sub conjunto de dados de uma temporada da NBA.

Em ambas os sub conjuntos sofrem as seguintes operações:
    
. Redução de Dimensão: Os atributos 'PLAYER_NAME', 'EVENT_TYPE' e 'TEAM_NAME' são retirados por serem redundantes.

. Dummy Coding: Aplicado nos seguintes atrobutos categóricos  'GRID_TYPE', 'ACTION_TYPE', 'SHOT_TYPE', 'SHOT_ZONE_BASIC', 
      'SHOT_ZONE_AREA', 'SHOT_ZONE_RANGE', 'HTM', 'VTM' e 'SEASON_ID'.

. Train/Test split: Os conjuntos de Treino e Teste foram criados.

. Checagem e tratamento de atributos vom variância igual a zero: Foram calculadas as variâncias de todos os atributos e os com valor 0 foram retirados das bases de Treino e Teste.

. Normalização: Os dados das bases de Treino e Teste foram normalizados.


    

