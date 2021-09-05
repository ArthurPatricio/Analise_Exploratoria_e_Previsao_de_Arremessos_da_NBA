# Analise Exploratoria e Previsao de Arremessos da NBA

# Objetivo


* Este projeto visa primeiramente, realizar uma análise exploratória dos dados obtidos das últimas 6 temporadas regulares da NBA (2015-16 a 2020-21) e treinar diferentes modelos de machine learning com o intuito de prever se um arremesso é bem-sucedido ou não.
* Os dados foram obtidos através da API da NBA, o script 'get_players_shot_charts.ipynb' criado e a planilha com os ID's dos jogadores pode ser encontrados em: (https://github.com/ArthurPatricio/Analise_Exploratoria_e_Previsao_de_Arremessos_da_NBA) 


{
  "nbformat": 4,
  "nbformat_minor": 2,
  "metadata": {
    "orig_nbformat": 4,
    "language_info": {
      "name": "python",
      "version": "3.8.6",
      "mimetype": "text/x-python",
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "pygments_lexer": "ipython3",
      "nbconvert_exporter": "python",
      "file_extension": ".py"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3.8.6 64-bit"
    },
    "interpreter": {
      "hash": "f499c09746ed89bb16a1ef7cbb581cc63f8572953d5a366b82ca25faa5a00be4"
    },
    "colab": {
      "name": "WORK_BOOK_V2.ipynb",
      "provenance": []
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "#Import Libs\r\n",
        "\r\n",
        "import pandas as pd\r\n",
        "import numpy as np\r\n",
        "import matplotlib.pyplot as plt\r\n",
        "import seaborn as sns\r\n",
        "import os\r\n",
        "import missingno as msno\r\n",
        "from pandas_profiling import ProfileReport\r\n",
        "import plotly.express as px\r\n",
        "import matplotlib as mpl\r\n",
        "import time\r\n",
        "from matplotlib.patches import Circle, Rectangle, Arc, ConnectionPatch\r\n",
        "from matplotlib.patches import Polygon\r\n",
        "from matplotlib.collections import PatchCollection\r\n",
        "from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm\r\n",
        "from matplotlib.path import Path\r\n",
        "from matplotlib.patches import PathPatch"
      ],
      "outputs": [],
      "metadata": {
        "id": "GLUma1CdaqQ6"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "os.chdir(r\"C:\\Users\\arthu\\Códigos e Relatórios\\DM 2021.1\\Trabalho\")"
      ],
      "outputs": [],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 180
        },
        "id": "ZUDGGKvJaqQ7",
        "outputId": "843c50b9-d821-44e1-bfe8-5d1ec2b12727"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "#Read NBA Shots excel files\r\n",
        "\r\n",
        "nba_shots_2020_21 = pd.read_excel('nba_shots_2020-21.xlsx', engine='openpyxl')\r\n",
        "nba_shots_2019_20 = pd.read_excel('nba_shots_2019-20.xlsx', engine='openpyxl')\r\n",
        "nba_shots_2018_19 = pd.read_excel('nba_shots_2018-19.xlsx', engine='openpyxl')\r\n",
        "nba_shots_2017_18 = pd.read_excel('nba_shots_2017-18.xlsx', engine='openpyxl')\r\n",
        "nba_shots_2016_17 = pd.read_excel('nba_shots_2016-17.xlsx', engine='openpyxl')\r\n",
        "nba_shots_2015_16 = pd.read_excel('nba_shots_2015-16.xlsx', engine='openpyxl')"
      ],
      "outputs": [],
      "metadata": {
        "id": "EJSsgCRhaqQ8"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "#Drop \"Unnamed: 0\" column, Add \"SEASON_ID\" column in nba_shots_2020_21\r\n",
        "\r\n",
        "nba_shots_2020_21.drop(['Unnamed: 0'], axis=1, inplace=True)\r\n",
        "nba_shots_2020_21['SEASON_ID'] = '2020-21'\r\n",
        "nba_shots_2020_21.head()"
      ],
      "outputs": [],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 309
        },
        "id": "PUkSPZZhaqQ9",
        "outputId": "3b9da2df-fa02-4330-c403-e102c983fc54"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "#Drop \"Unnamed: 0\" column, Add \"SEASON_ID\" column in nba_shots_2019_20\r\n",
        "\r\n",
        "nba_shots_2019_20.drop(['Unnamed: 0'], axis=1, inplace=True)\r\n",
        "nba_shots_2019_20['SEASON_ID'] = '2019-20'\r\n",
        "nba_shots_2019_20.head()"
      ],
      "outputs": [],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 309
        },
        "id": "YeZ1-SAHaqQ-",
        "outputId": "1e07d56d-f618-4820-8e47-ecd48887b95a"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "#Drop \"Unnamed: 0\" column, Add \"SEASON_ID\" column in nba_shots_2018_19\r\n",
        "\r\n",
        "nba_shots_2018_19.drop(['Unnamed: 0'], axis=1, inplace=True)\r\n",
        "nba_shots_2018_19['SEASON_ID'] = '2018-19'\r\n",
        "nba_shots_2018_19.head()"
      ],
      "outputs": [],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 309
        },
        "id": "Y0R2c_vNaqQ-",
        "outputId": "07c66164-4416-4924-915b-5db79c12d118"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "#Drop \"Unnamed: 0\" column, Add \"SEASON_ID\" column in nba_shots_2017_18\r\n",
        "\r\n",
        "nba_shots_2017_18.drop(['Unnamed: 0'], axis=1, inplace=True)\r\n",
        "nba_shots_2017_18['SEASON_ID'] = '2017-18'\r\n",
        "nba_shots_2017_18.head()"
      ],
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "#Drop \"Unnamed: 0\" column, Add \"SEASON_ID\" column in nba_shots_2016_17\r\n",
        "\r\n",
        "nba_shots_2016_17.drop(['Unnamed: 0'], axis=1, inplace=True)\r\n",
        "nba_shots_2016_17['SEASON_ID'] = '2016-17'\r\n",
        "nba_shots_2016_17.head()"
      ],
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "#Drop \"Unnamed: 0\" column, Add \"SEASON_ID\" column in nba_shots_2015_16\r\n",
        "\r\n",
        "nba_shots_2015_16.drop(['Unnamed: 0'], axis=1, inplace=True)\r\n",
        "nba_shots_2015_16['SEASON_ID'] = '2015-16'\r\n",
        "nba_shots_2015_16.head()"
      ],
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "#Create nba_shots as a concatenation of the 3 Dataframes from each reagular season\r\n",
        "\r\n",
        "nba_shots = pd.concat([nba_shots_2020_21, nba_shots_2019_20, nba_shots_2018_19, nba_shots_2017_18, nba_shots_2016_17, nba_shots_2015_16], sort=False)\r\n",
        "\r\n",
        "nba_shots.head()"
      ],
      "outputs": [],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 309
        },
        "id": "imiei5zFaqQ_",
        "outputId": "ed77f1d3-db8b-417d-81cb-3fdc58c0f300"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "#Get nba_shots dataframe shape\r\n",
        "\r\n",
        "nba_shots.shape"
      ],
      "outputs": [],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "LRuMpbdgaqRA",
        "outputId": "86d764d8-2047-471a-9a7a-6b82c17878fb"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "#Get nba_shots dataframe columns\r\n",
        "\r\n",
        "nba_shots.columns"
      ],
      "outputs": [],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "gCHDNp4waqRB",
        "outputId": "0f1fb9ce-86b5-487e-f63e-cfe4a32a1944"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "#Get nba_shots dataframe describe\r\n",
        "\r\n",
        "nba_shots.describe()"
      ],
      "outputs": [],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 317
        },
        "id": "Pv0oBW9uaqRB",
        "outputId": "b84f048b-ce7d-4b35-8803-8da4fb177923"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "#Get nba_shots dataframe info\r\n",
        "\r\n",
        "nba_shots.info()"
      ],
      "outputs": [],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Tng8tFdBaqRC",
        "outputId": "bffb4575-1e8d-4678-fddb-4faf490e0b4f"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "#Check for messing values in the nba_shots dataframe\r\n",
        "\r\n",
        "msno.matrix(nba_shots)"
      ],
      "outputs": [],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "3UMQQyXHaqRC",
        "outputId": "4cd05c17-2857-413d-da53-9401b8e41e26"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "#Generate and export as a .html file the Pandas Profile Report of the nba_shots dataframe\r\n",
        "\r\n",
        "profile_shots = ProfileReport(nba_shots, title ='nba_shots')\r\n",
        "profile_shots.to_file(\"nba_shots.html\")"
      ],
      "outputs": [],
      "metadata": {
        "id": "DeyWFj18aqRD",
        "outputId": "57b30765-0186-4d4e-ce88-111332cc455c"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "#Show Profile Report in this notebook\r\n",
        "\r\n",
        "profile_shots.to_notebook_iframe()"
      ],
      "outputs": [],
      "metadata": {
        "id": "pp-AtvjNaqRD"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "# Function to draw basketball court\r\n",
        "\r\n",
        "def create_court(ax, color):\r\n",
        "    \r\n",
        "    # Short corner 3PT lines\r\n",
        "    ax.plot([-220, -220], [0, 140], linewidth=2, color=color)\r\n",
        "    ax.plot([220, 220], [0, 140], linewidth=2, color=color)\r\n",
        "    \r\n",
        "    # 3PT Arc\r\n",
        "    ax.add_artist(mpl.patches.Arc((0, 140), 440, 315, theta1=0, theta2=180, facecolor='none', edgecolor=color, lw=2))\r\n",
        "    \r\n",
        "    # Lane and Key\r\n",
        "    ax.plot([-80, -80], [0, 190], linewidth=2, color=color)\r\n",
        "    ax.plot([80, 80], [0, 190], linewidth=2, color=color)\r\n",
        "    ax.plot([-60, -60], [0, 190], linewidth=2, color=color)\r\n",
        "    ax.plot([60, 60], [0, 190], linewidth=2, color=color)\r\n",
        "    ax.plot([-80, 80], [190, 190], linewidth=2, color=color)\r\n",
        "    ax.add_artist(mpl.patches.Circle((0, 190), 60, facecolor='none', edgecolor=color, lw=2))\r\n",
        "    \r\n",
        "    # Rim\r\n",
        "    ax.add_artist(mpl.patches.Circle((0, 60), 15, facecolor='none', edgecolor=color, lw=2))\r\n",
        "    \r\n",
        "    # Backboard\r\n",
        "    ax.plot([-30, 30], [40, 40], linewidth=2, color=color)\r\n",
        "    \r\n",
        "    # Remove ticks\r\n",
        "    ax.set_xticks([])\r\n",
        "    ax.set_yticks([])\r\n",
        "    \r\n",
        "    # Set axis limits\r\n",
        "    ax.set_xlim(-250, 250)\r\n",
        "    ax.set_ylim(0, 470)\r\n",
        "\r\n",
        "    # General plot parameters\r\n",
        "    mpl.rcParams['font.family'] = 'Avenir'\r\n",
        "    mpl.rcParams['font.size'] = 18\r\n",
        "    mpl.rcParams['axes.linewidth'] = 2\r\n",
        "    \r\n",
        "    return ax\r\n"
      ],
      "outputs": [],
      "metadata": {
        "id": "VqgmErDUaqRD"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "# JAMES HARDEN 2020-21 REGULAR SEASON SHOTS\r\n",
        "\r\n",
        "# Create figure and axes\r\n",
        "fig = plt.figure(figsize=(10, 9))\r\n",
        "ax = fig.add_axes([0, 0, 1, 1])\r\n",
        "\r\n",
        "# Draw court\r\n",
        "ax = create_court(ax, 'black')\r\n",
        "\r\n",
        "# Shots Scatter Plots\r\n",
        "ax.scatter(nba_shots[(nba_shots['SEASON_ID'] == '2020-21') & (nba_shots['SHOT_MADE_FLAG']==1) & (nba_shots['PLAYER_NAME'] == 'James Harden')]['LOC_X'],\r\n",
        "            nba_shots[(nba_shots['SEASON_ID'] == '2020-21') & (nba_shots['SHOT_MADE_FLAG']==1) & (nba_shots['PLAYER_NAME'] == 'James Harden')]['LOC_Y'] +60, marker = \"o\", color = \"Green\")\r\n",
        "\r\n",
        "ax.scatter(nba_shots[(nba_shots['SEASON_ID'] == '2020-21') & (nba_shots['SHOT_MADE_FLAG']==0) & (nba_shots['PLAYER_NAME'] == 'James Harden')]['LOC_X'],\r\n",
        "            nba_shots[(nba_shots['SEASON_ID'] == '2020-21') & (nba_shots['SHOT_MADE_FLAG']==0) & (nba_shots['PLAYER_NAME'] == 'James Harden')]['LOC_Y'] +60, marker = \"x\", color = \"Red\")\r\n",
        "\r\n",
        "plt.title('JAMES HARDEN 2020-21 REGULAR SEASON SHOTS', fontsize = 20)\r\n",
        "plt.show()"
      ],
      "outputs": [],
      "metadata": {
        "id": "Zx-EKedWaqRE"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "# SEPHEN CURRY 2020-21 REGULAR SEASON SHOTS\r\n",
        "\r\n",
        "# Create figure and axes\r\n",
        "fig = plt.figure(figsize=(10, 9))\r\n",
        "ax = fig.add_axes([0, 0, 1, 1])\r\n",
        "\r\n",
        "# Draw court\r\n",
        "ax = create_court(ax, 'black')\r\n",
        "\r\n",
        "# Plot scatter of shots\r\n",
        "ax.scatter(nba_shots[(nba_shots['SEASON_ID'] == '2020-21') & (nba_shots['SHOT_MADE_FLAG']==1) & (nba_shots['PLAYER_NAME'] == 'Stephen Curry')]['LOC_X'],\r\n",
        "            nba_shots[(nba_shots['SEASON_ID'] == '2020-21') & (nba_shots['SHOT_MADE_FLAG']==1) & (nba_shots['PLAYER_NAME'] == 'Stephen Curry')]['LOC_Y'] +60, marker = \"o\", color = \"Green\")\r\n",
        "\r\n",
        "ax.scatter(nba_shots[(nba_shots['SEASON_ID'] == '2020-21') & (nba_shots['SHOT_MADE_FLAG']==0) & (nba_shots['PLAYER_NAME'] == 'Stephen Curry')]['LOC_X'],\r\n",
        "            nba_shots[(nba_shots['SEASON_ID'] == '2020-21') & (nba_shots['SHOT_MADE_FLAG']==0) & (nba_shots['PLAYER_NAME'] == 'Stephen Curry')]['LOC_Y'] +60, marker = \"x\", color = \"Red\")\r\n",
        "\r\n",
        "plt.title('STEPHEN CURRY 2020-21 REGULAR SEASON SHOTS', fontsize = 20)\r\n",
        "plt.show()"
      ],
      "outputs": [],
      "metadata": {
        "id": "Ya09Ht6daqRE"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "# NIKOLA JOKIC 2020-21 REGULAR SEASON SHOTS\r\n",
        "\r\n",
        "# Create figure and axes\r\n",
        "fig = plt.figure(figsize=(10, 9))\r\n",
        "ax = fig.add_axes([0, 0, 1, 1])\r\n",
        "\r\n",
        "# Draw court\r\n",
        "ax = create_court(ax, 'black')\r\n",
        "\r\n",
        "# Plot scatter of shots\r\n",
        "ax.scatter(nba_shots[(nba_shots['SEASON_ID'] == '2020-21') & (nba_shots['SHOT_MADE_FLAG']==1) & (nba_shots['PLAYER_NAME'] == 'Nikola Jokic')]['LOC_X'],\r\n",
        "            nba_shots[(nba_shots['SEASON_ID'] == '2020-21') & (nba_shots['SHOT_MADE_FLAG']==1) & (nba_shots['PLAYER_NAME'] == 'Nikola Jokic')]['LOC_Y'] +60, marker = \"o\", color = \"Green\")\r\n",
        "\r\n",
        "ax.scatter(nba_shots[(nba_shots['SEASON_ID'] == '2020-21') & (nba_shots['SHOT_MADE_FLAG']==0) & (nba_shots['PLAYER_NAME'] == 'Nikola Jokic')]['LOC_X'],\r\n",
        "            nba_shots[(nba_shots['SEASON_ID'] == '2020-21') & (nba_shots['SHOT_MADE_FLAG']==0) & (nba_shots['PLAYER_NAME'] == 'Nikola Jokic')]['LOC_Y'] +60, marker = \"x\", color = \"Red\")\r\n",
        "\r\n",
        "plt.title('NIKOLA JOKIC 2020-21 REGULAR SEASON SHOTS', fontsize = 20)\r\n",
        "plt.show()"
      ],
      "outputs": [],
      "metadata": {
        "id": "54Go_BvuaqRE"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "# 2020-21 REGULAR SEASON MADE SHOTS\r\n",
        "\r\n",
        "# Create figure and axes\r\n",
        "fig = plt.figure(figsize=(10, 9))\r\n",
        "ax = fig.add_axes([0, 0, 1, 1])\r\n",
        "\r\n",
        "# Draw court\r\n",
        "ax = create_court(ax, 'black')\r\n",
        "\r\n",
        "# Plot scatter of shots\r\n",
        "ax.hexbin(nba_shots[(nba_shots['SEASON_ID'] == '2020-21') & (nba_shots['SHOT_MADE_FLAG']==1)]['LOC_X'],\r\n",
        "            nba_shots[(nba_shots['SEASON_ID'] == '2020-21') & (nba_shots['SHOT_MADE_FLAG']==1)]['LOC_Y'] +60, gridsize=(30, 30), extent=(-300, 300, 0, 940), bins='log', cmap='Greens')\r\n",
        "\r\n",
        "plt.title('2020-21 REGULAR SEASON MADE SHOTS', fontsize = 20)\r\n",
        "plt.show()"
      ],
      "outputs": [],
      "metadata": {
        "id": "NhIDe7dDaqRF"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "# 2020-21 REGULAR SEASON SHOTS MADE PER ZONE AREA\r\n",
        "\r\n",
        "# Create figure and axes\r\n",
        "fig = plt.figure(figsize=(20, 9))\r\n",
        "ax = fig.add_axes([0, 0, 1, 1])\r\n",
        "\r\n",
        "# Draw court\r\n",
        "ax = create_court(ax, 'black')\r\n",
        "\r\n",
        "# Plot scatter of shots\r\n",
        "sns.scatterplot(nba_shots[(nba_shots['SEASON_ID'] == '2020-21') & (nba_shots['SHOT_MADE_FLAG']==1)]['LOC_X'],\r\n",
        "            nba_shots[(nba_shots['SEASON_ID'] == '2020-21') & (nba_shots['SHOT_MADE_FLAG']==1)]['LOC_Y'] + 60, hue = nba_shots[(nba_shots['SEASON_ID'] == '2020-21') & (nba_shots['SHOT_MADE_FLAG']==1)]['SHOT_ZONE_AREA'])\r\n",
        "\r\n",
        "plt.title('2020-21 REGULAR SEASON SHOTS MADE PER ZONE AREA', fontsize = 20)\r\n",
        "plt.show()"
      ],
      "outputs": [],
      "metadata": {
        "id": "Qk45C-R4aqRF"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "# 2020-21 REGULAR SEASON SHOTS MADE PER ZONE RANGE\r\n",
        "\r\n",
        "# Create figure and axes\r\n",
        "fig = plt.figure(figsize=(20, 9))\r\n",
        "ax = fig.add_axes([0, 0, 1, 1])\r\n",
        "\r\n",
        "# Draw court\r\n",
        "ax = create_court(ax, 'black')\r\n",
        "\r\n",
        "# Plot scatter of shots\r\n",
        "sns.scatterplot(nba_shots[(nba_shots['SEASON_ID'] == '2020-21') & (nba_shots['SHOT_MADE_FLAG']==1)]['LOC_X'],\r\n",
        "            nba_shots[(nba_shots['SEASON_ID'] == '2020-21') & (nba_shots['SHOT_MADE_FLAG']==1)]['LOC_Y'] + 60, hue = nba_shots[(nba_shots['SEASON_ID'] == '2020-21') & (nba_shots['SHOT_MADE_FLAG']==1)]['SHOT_ZONE_RANGE'])\r\n",
        "\r\n",
        "plt.title('2020-21 REGULAR SEASON SHOTS MADE PER ZONE RANGE', fontsize = 20)\r\n",
        "plt.show()"
      ],
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "# 2020-21 REGULAR SEASON SHOTS MADE PER ZONE AREA (BASIC)\r\n",
        "\r\n",
        "# Create figure and axes\r\n",
        "fig = plt.figure(figsize=(20, 9))\r\n",
        "ax = fig.add_axes([0, 0, 1, 1])\r\n",
        "\r\n",
        "# Draw court\r\n",
        "ax = create_court(ax, 'black')\r\n",
        "\r\n",
        "# Plot scatter of shots\r\n",
        "sns.scatterplot(nba_shots[(nba_shots['SEASON_ID'] == '2020-21') & (nba_shots['SHOT_MADE_FLAG']==1)]['LOC_X'],\r\n",
        "            nba_shots[(nba_shots['SEASON_ID'] == '2020-21') & (nba_shots['SHOT_MADE_FLAG']==1)]['LOC_Y'] + 60, hue = nba_shots[(nba_shots['SEASON_ID'] == '2020-21') & (nba_shots['SHOT_MADE_FLAG']==1)]['SHOT_ZONE_BASIC'])\r\n",
        "\r\n",
        "plt.title('2020-21 REGULAR SEASON SHOTS MADE PER ZONE AREA (BASIC)', fontsize = 20)\r\n",
        "plt.show()"
      ],
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "# 2019-20 REGULAR SEASON MADE SHOTS\r\n",
        "\r\n",
        "# Create figure and axes\r\n",
        "fig = plt.figure(figsize=(10, 9))\r\n",
        "ax = fig.add_axes([0, 0, 1, 1])\r\n",
        "\r\n",
        "# Draw court\r\n",
        "ax = create_court(ax, 'black')\r\n",
        "\r\n",
        "# Plot scatter of shots\r\n",
        "ax.hexbin(nba_shots[(nba_shots['SEASON_ID'] == '2019-20') & (nba_shots['SHOT_MADE_FLAG']==1)]['LOC_X'],\r\n",
        "            nba_shots[(nba_shots['SEASON_ID'] == '2019-20') & (nba_shots['SHOT_MADE_FLAG']==1)]['LOC_Y'] +60, gridsize=(30, 30), extent=(-300, 300, 0, 940), bins='log', cmap='Greens')\r\n",
        "\r\n",
        "plt.title('2019-20 REGULAR SEASON MADE SHOTS', fontsize = 20)\r\n",
        "plt.show()"
      ],
      "outputs": [],
      "metadata": {
        "id": "Nw2aPgbMaqRF"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "# 2019-20 REGULAR SEASON SHOTS PER ZONE AREA\r\n",
        "\r\n",
        "# Create figure and axes\r\n",
        "fig = plt.figure(figsize=(20, 9))\r\n",
        "ax = fig.add_axes([0, 0, 1, 1])\r\n",
        "\r\n",
        "# Draw court\r\n",
        "ax = create_court(ax, 'black')\r\n",
        "\r\n",
        "# Plot scatter of shots\r\n",
        "sns.scatterplot(nba_shots[(nba_shots['SEASON_ID'] == '2019-20') & (nba_shots['SHOT_MADE_FLAG']==1)]['LOC_X'],\r\n",
        "            nba_shots[(nba_shots['SEASON_ID'] == '2019-20') & (nba_shots['SHOT_MADE_FLAG']==1)]['LOC_Y'] + 60, hue = nba_shots[(nba_shots['SEASON_ID'] == '2019-20') & (nba_shots['SHOT_MADE_FLAG']==1)]['SHOT_ZONE_AREA'])\r\n",
        "\r\n",
        "plt.title('2019-20 REGULAR SEASON SHOTS PER ZONE AREA', fontsize = 20)\r\n",
        "plt.show()"
      ],
      "outputs": [],
      "metadata": {
        "id": "RaaH0MpMaqRG"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "# 2018-19 REGULAR SEASON MADE SHOTS\r\n",
        "\r\n",
        "# Create figure and axes\r\n",
        "fig = plt.figure(figsize=(10, 9))\r\n",
        "ax = fig.add_axes([0, 0, 1, 1])\r\n",
        "\r\n",
        "# Draw court\r\n",
        "ax = create_court(ax, 'black')\r\n",
        "\r\n",
        "# Plot scatter of shots\r\n",
        "ax.hexbin(nba_shots[(nba_shots['SEASON_ID'] == '2018-19') & (nba_shots['SHOT_MADE_FLAG']==1)]['LOC_X'],\r\n",
        "            nba_shots[(nba_shots['SEASON_ID'] == '2018-19') & (nba_shots['SHOT_MADE_FLAG']==1)]['LOC_Y'] +60, gridsize=(30, 30), extent=(-300, 300, 0, 940), bins='log', cmap='Greens')\r\n",
        "\r\n",
        "plt.title('2018-19 REGULAR SEASON MADE SHOTS', fontsize = 20)\r\n",
        "plt.show()"
      ],
      "outputs": [],
      "metadata": {
        "id": "LyHnvvFDaqRG"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "# 2018-19 REGULAR SEASON SHOTS PER ZONE AREA\r\n",
        "\r\n",
        "# Create figure and axes\r\n",
        "fig = plt.figure(figsize=(20, 9))\r\n",
        "ax = fig.add_axes([0, 0, 1, 1])\r\n",
        "\r\n",
        "# Draw court\r\n",
        "ax = create_court(ax, 'black')\r\n",
        "\r\n",
        "# Plot scatter of shots\r\n",
        "sns.scatterplot(nba_shots[(nba_shots['SEASON_ID'] == '2018-19') & (nba_shots['SHOT_MADE_FLAG']==1)]['LOC_X'],\r\n",
        "            nba_shots[(nba_shots['SEASON_ID'] == '2018-19') & (nba_shots['SHOT_MADE_FLAG']==1)]['LOC_Y'] + 60, hue = nba_shots[(nba_shots['SEASON_ID'] == '2018-19') & (nba_shots['SHOT_MADE_FLAG']==1)]['SHOT_ZONE_AREA'])\r\n",
        "\r\n",
        "plt.title('2018-19 REGULAR SEASON SHOTS PER ZONE AREA', fontsize = 20)\r\n",
        "plt.show()"
      ],
      "outputs": [],
      "metadata": {
        "id": "12IRqkswaqRG"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "# 2017-18 REGULAR SEASON MADE SHOTS\r\n",
        "\r\n",
        "# Create figure and axes\r\n",
        "fig = plt.figure(figsize=(10, 9))\r\n",
        "ax = fig.add_axes([0, 0, 1, 1])\r\n",
        "\r\n",
        "# Draw court\r\n",
        "ax = create_court(ax, 'black')\r\n",
        "\r\n",
        "# Plot scatter of shots\r\n",
        "ax.hexbin(nba_shots[(nba_shots['SEASON_ID'] == '2017-18') & (nba_shots['SHOT_MADE_FLAG']==1)]['LOC_X'],\r\n",
        "            nba_shots[(nba_shots['SEASON_ID'] == '2017-18') & (nba_shots['SHOT_MADE_FLAG']==1)]['LOC_Y'] +60, gridsize=(30, 30), extent=(-300, 300, 0, 940), bins='log', cmap='Greens')\r\n",
        "\r\n",
        "plt.title('2017-18 REGULAR SEASON MADE SHOTS', fontsize = 20)\r\n",
        "plt.show()"
      ],
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "# 2016-17 REGULAR SEASON MADE SHOTS\r\n",
        "\r\n",
        "# Create figure and axes\r\n",
        "fig = plt.figure(figsize=(10, 9))\r\n",
        "ax = fig.add_axes([0, 0, 1, 1])\r\n",
        "\r\n",
        "# Draw court\r\n",
        "ax = create_court(ax, 'black')\r\n",
        "\r\n",
        "# Plot scatter of shots\r\n",
        "ax.hexbin(nba_shots[(nba_shots['SEASON_ID'] == '2016-17') & (nba_shots['SHOT_MADE_FLAG']==1)]['LOC_X'],\r\n",
        "            nba_shots[(nba_shots['SEASON_ID'] == '2016-17') & (nba_shots['SHOT_MADE_FLAG']==1)]['LOC_Y'] +60, gridsize=(30, 30), extent=(-300, 300, 0, 940), bins='log', cmap='Greens')\r\n",
        "\r\n",
        "plt.title('2016-17 REGULAR SEASON MADE SHOTS', fontsize = 20)\r\n",
        "plt.show()"
      ],
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "# 2015-16 REGULAR SEASON MADE SHOTS\r\n",
        "\r\n",
        "# Create figure and axes\r\n",
        "fig = plt.figure(figsize=(10, 9))\r\n",
        "ax = fig.add_axes([0, 0, 1, 1])\r\n",
        "\r\n",
        "# Draw court\r\n",
        "ax = create_court(ax, 'black')\r\n",
        "\r\n",
        "# Plot scatter of shots\r\n",
        "ax.hexbin(nba_shots[(nba_shots['SEASON_ID'] == '2015-16') & (nba_shots['SHOT_MADE_FLAG']==1)]['LOC_X'],\r\n",
        "            nba_shots[(nba_shots['SEASON_ID'] == '2015-16') & (nba_shots['SHOT_MADE_FLAG']==1)]['LOC_Y'] +60, gridsize=(30, 30), extent=(-300, 300, 0, 940), bins='log', cmap='Greens')\r\n",
        "\r\n",
        "plt.title('2015-16 REGULAR SEASON MADE SHOTS', fontsize = 20)\r\n",
        "plt.show()"
      ],
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "# SHOT DISTANCE DISTRIBUTION PLOT\r\n",
        "\r\n",
        "plt.figure(figsize=(20,12))\r\n",
        "fig1 = sns.histplot(data=nba_shots, x='SHOT_DISTANCE', hue = 'SHOT_TYPE')\r\n",
        "fig1.set_xlabel('SHOT_DISTANCE', fontsize=20)\r\n",
        "fig1.set_ylabel('COUNT', fontsize=20)\r\n",
        "fig1.tick_params(labelsize=15)\r\n",
        "plt.title('SHOT DISTANCE DISTRIBUTION', fontsize = 20)\r\n",
        "plt.xlim(0,40)\r\n",
        "plt.show()"
      ],
      "outputs": [],
      "metadata": {
        "id": "FaLDnIvRaqRG"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "# SHOT TYPE BAR PLOT\r\n",
        "\r\n",
        "plt.figure(figsize=(20,12))\r\n",
        "fig2 = sns.countplot(data=nba_shots, x='SHOT_TYPE', palette = 'husl')\r\n",
        "fig2.set_xlabel('SHOT_TYPE', fontsize=20)\r\n",
        "fig2.set_ylabel('COUNT', fontsize=20)\r\n",
        "fig2.tick_params(labelsize=20)\r\n",
        "plt.title('SHOT TYPE', fontsize = 20)\r\n",
        "for p in fig2.patches:\r\n",
        "    txt = str(p.get_height().round(2))\r\n",
        "    txt_x = p.get_x() \r\n",
        "    txt_y = p.get_height()\r\n",
        "    fig2.text(txt_x,txt_y,txt)\r\n",
        "plt.show()"
      ],
      "outputs": [],
      "metadata": {
        "id": "rcKFcJTbaqRG"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "# SHOT TYPE (MADE/MISSED) BAR PLOT\r\n",
        "\r\n",
        "plt.figure(figsize=(20,12))\r\n",
        "fig3 = sns.countplot(data=nba_shots, x='SHOT_TYPE', palette = 'husl', hue = 'EVENT_TYPE')\r\n",
        "fig3.set_xlabel('SHOT_TYPE', fontsize=20)\r\n",
        "fig3.set_ylabel('COUNT', fontsize=20)\r\n",
        "fig3.tick_params(labelsize=20)\r\n",
        "plt.title('SHOT TYPE', fontsize = 20)\r\n",
        "for p in fig3.patches:\r\n",
        "    txt = str(p.get_height().round(2))\r\n",
        "    txt_x = p.get_x() \r\n",
        "    txt_y = p.get_height()\r\n",
        "    fig3.text(txt_x,txt_y,txt)\r\n",
        "plt.show()"
      ],
      "outputs": [],
      "metadata": {
        "id": "HgNhjvUaaqRH"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "# SHOT ACTION TYPE BAR PLOT\r\n",
        "\r\n",
        "plt.figure(figsize=(45,12))\r\n",
        "fig4 = sns.countplot(data=nba_shots, x='ACTION_TYPE', palette = 'husl') \r\n",
        "fig4.set_xlabel('ACTION_TYPE', fontsize=20)\r\n",
        "fig4.set_ylabel('COUNT', fontsize=20)\r\n",
        "fig4.tick_params(labelsize=20)\r\n",
        "fig4.tick_params(axis = 'x', rotation = 90)\r\n",
        "plt.title('SHOT ACTION TYPE', fontsize = 20)\r\n",
        "for p in fig4.patches:\r\n",
        "    txt = str(p.get_height().round(2))\r\n",
        "    txt_x = p.get_x() \r\n",
        "    txt_y = p.get_height()\r\n",
        "    fig4.text(txt_x,txt_y,txt)\r\n",
        "plt.show()"
      ],
      "outputs": [],
      "metadata": {
        "id": "QcIJau6zaqRH"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "# Import Libs\r\n",
        "\r\n",
        "from sklearn.feature_selection import VarianceThreshold\r\n",
        "from sklearn.model_selection import train_test_split\r\n",
        "from sklearn.svm import SVC\r\n",
        "from sklearn.svm import LinearSVC\r\n",
        "from sklearn.tree import DecisionTreeClassifier\r\n",
        "from sklearn import tree\r\n",
        "from sklearn.preprocessing import StandardScaler\r\n",
        "from sklearn.model_selection import GridSearchCV\r\n",
        "from sklearn.metrics import classification_report\r\n",
        "from sklearn.ensemble import RandomForestClassifier\r\n",
        "from sklearn.ensemble import GradientBoostingClassifier\r\n",
        "import xgboost as xgb"
      ],
      "outputs": [],
      "metadata": {
        "id": "ltXTB0vpaqRH"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "#Create player sub dataset"
      ],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "# Due to the large amount of the dataset, everything past this point you be done per player. The function below creates a sub dataset from nba_shots with the data form the chosen player. \r\n",
        "\r\n",
        "def choose_player (player_name):\r\n",
        "    player_shots = nba_shots[nba_shots['PLAYER_NAME'] == player_name]\r\n",
        "    print(player_shots.head())\r\n",
        "    \r\n",
        "    # Dimensional Reduction: Columns PLAYER_ID and PLAYER_NAME carry the same type of information, PLAYER_NAME is going to be droped. \r\n",
        "    # The same happens to columns EVENT_TYPE and SHOT_MADE_FLAG, EVENT_TYPE is going to be droped.\r\n",
        "    # It also happens for TEAM_ID and TEAM_NAME, TEAM_NAME is going to be droped.\r\n",
        "\r\n",
        "    nba_shots_ml = player_shots.drop(['PLAYER_NAME', 'EVENT_TYPE', 'TEAM_NAME'], axis = 1)\r\n",
        "    \r\n",
        "    # Apply Dummy Coding to the categorial attributes of the dataset\r\n",
        "\r\n",
        "    categorical_columns = ['GRID_TYPE', 'ACTION_TYPE', 'SHOT_TYPE', 'SHOT_ZONE_BASIC', \r\n",
        "    'SHOT_ZONE_AREA', 'SHOT_ZONE_RANGE', 'HTM', 'VTM', 'SEASON_ID']\r\n",
        "\r\n",
        "    for i in categorical_columns:\r\n",
        "\r\n",
        "        nba_shots_ml = pd.get_dummies(nba_shots_ml, columns=[i], drop_first=True)\r\n",
        "\r\n",
        "    #Train/Test split\r\n",
        "\r\n",
        "    X = nba_shots_ml.loc[:, nba_shots_ml.columns != 'SHOT_MADE_FLAG']\r\n",
        "    y = nba_shots_ml['SHOT_MADE_FLAG']\r\n",
        "    X_train, X_test, y_train, y_test = train_test_split(X, y, \r\n",
        "                                                        test_size= 0.3, \r\n",
        "                                                        random_state = 100, \r\n",
        "                                                        stratify = y,\r\n",
        "                                                        )\r\n",
        "\r\n",
        "    # Check columns with variance equal to zero and drop them\r\n",
        "\r\n",
        "    zero_var_filter = VarianceThreshold()\r\n",
        "    X_train = zero_var_filter.fit_transform(X_train)\r\n",
        "    X_test = zero_var_filter.transform(X_test)\r\n",
        "    print('X_train e X_test possuíam', (zero_var_filter.variances_ == 0).sum(), 'atributo(s) com variância igual a zero')\r\n",
        "\r\n",
        "    print('X_train:', X_train.shape)\r\n",
        "    print('X_test:', X_test.shape)\r\n",
        "    print('y_train:', y_train.shape)\r\n",
        "    print('y_test:', y_test.shape)\r\n",
        "\r\n",
        "    # Normalize the data\r\n",
        "\r\n",
        "    scaler = StandardScaler().fit(X_train)\r\n",
        "    X_train = scaler.transform(X_train)\r\n",
        "    X_test = scaler.transform(X_test)\r\n",
        "\r\n",
        "    return X_train, X_test, y_train, y_test\r\n",
        "\r\n",
        "X_train, X_test, y_train, y_test = choose_player('Stephen Curry')"
      ],
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": [
        "#Create season sub dataset"
      ],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "# Due to the large amount of the dataset, everything past this point you be done per season. The function below creates a sub dataset from nba_shots with the data form the chosen season. \r\n",
        "\r\n",
        "def choose_season (season_id):\r\n",
        "    season_shots = nba_shots[nba_shots['SEASON_ID'] == season_id]\r\n",
        "    print(season_shots.head())\r\n",
        "    \r\n",
        "    # Dimensional Reduction: Columns PLAYER_ID and PLAYER_NAME carry the same type of information, PLAYER_NAME is going to be droped. \r\n",
        "    # The same happens to columns EVENT_TYPE and SHOT_MADE_FLAG, EVENT_TYPE is going to be droped.\r\n",
        "    # It also happens for TEAM_ID and TEAM_NAME, TEAM_NAME is going to be droped.\r\n",
        "\r\n",
        "    nba_shots_ml = season_shots.drop(['PLAYER_NAME', 'EVENT_TYPE', 'TEAM_NAME'], axis = 1)\r\n",
        "    \r\n",
        "    # Apply Dummy Coding to the categorial attributes of the dataset\r\n",
        "\r\n",
        "    categorical_columns = ['GRID_TYPE', 'ACTION_TYPE', 'SHOT_TYPE', 'SHOT_ZONE_BASIC', \r\n",
        "    'SHOT_ZONE_AREA', 'SHOT_ZONE_RANGE', 'HTM', 'VTM', 'SEASON_ID']\r\n",
        "\r\n",
        "    for i in categorical_columns:\r\n",
        "\r\n",
        "        nba_shots_ml = pd.get_dummies(nba_shots_ml, columns=[i], drop_first=True)\r\n",
        "\r\n",
        "    #Train/Test split\r\n",
        "\r\n",
        "    X = nba_shots_ml.loc[:, nba_shots_ml.columns != 'SHOT_MADE_FLAG']\r\n",
        "    y = nba_shots_ml['SHOT_MADE_FLAG']\r\n",
        "    X_train, X_test, y_train, y_test = train_test_split(X, y, \r\n",
        "                                                        test_size= 0.2, \r\n",
        "                                                        random_state = 100, \r\n",
        "                                                        #stratify = y\r\n",
        "                                                        )\r\n",
        "\r\n",
        "    # Check columns with variance equal to zero and drop them\r\n",
        "\r\n",
        "    zero_var_filter = VarianceThreshold()\r\n",
        "    X_train = zero_var_filter.fit_transform(X_train)\r\n",
        "    X_test = zero_var_filter.transform(X_test)\r\n",
        "    print('X_train e X_test possuíam', (zero_var_filter.variances_ == 0).sum(), 'atributo(s) com variância igual a zero')\r\n",
        "\r\n",
        "    print('X_train:', X_train.shape)\r\n",
        "    print('X_test:', X_test.shape)\r\n",
        "    print('y_train:', y_train.shape)\r\n",
        "    print('y_test:', y_test.shape)\r\n",
        "\r\n",
        "    # Normalize the data\r\n",
        "\r\n",
        "    scaler = StandardScaler().fit(X_train)\r\n",
        "    X_train = scaler.transform(X_train)\r\n",
        "    X_test = scaler.transform(X_test)\r\n",
        "\r\n",
        "    return X_train, X_test, y_train, y_test\r\n",
        "\r\n",
        "X_train, X_test, y_train, y_test = choose_season('2020-21')"
      ],
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": [
        "#SVM"
      ],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "# Train and predict SVM model\r\n",
        "\r\n",
        "def train_SVM(X_train, y_train):\r\n",
        "  model = SVC(random_state=100)\r\n",
        "  model.fit(X_train, y_train)\r\n",
        "  return model\r\n",
        "\r\n",
        "model_SVM = train_SVM(X_train, y_train)\r\n",
        "y_pred_SVM = model_SVM.predict(X_test)"
      ],
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "# Evaluate SVM model\r\n",
        "\r\n",
        "def evaluate_SVM(X_test, y_test):\r\n",
        "\r\n",
        "  # Acurácia\r\n",
        "  from sklearn.metrics import accuracy_score\r\n",
        "  accuracy = accuracy_score(y_test, y_pred_SVM)\r\n",
        "  print('Acurácia: ', accuracy)\r\n",
        "\r\n",
        "  # Kappa\r\n",
        "  from sklearn.metrics import cohen_kappa_score\r\n",
        "  kappa = cohen_kappa_score(y_test, y_pred_SVM)\r\n",
        "  print('Kappa: ', kappa)\r\n",
        "\r\n",
        "  # F1\r\n",
        "  from sklearn.metrics import f1_score\r\n",
        "  f1 = f1_score(y_test, y_pred_SVM)\r\n",
        "  print('F1: ', f1)\r\n",
        "\r\n",
        "  # Matriz de confusão\r\n",
        "  from sklearn.metrics import confusion_matrix\r\n",
        "  confMatrix = confusion_matrix(y_pred_SVM, y_test)\r\n",
        "\r\n",
        "  plt.figure(figsize=(20,12))\r\n",
        "  ax = plt.subplot()\r\n",
        "  sns.heatmap(confMatrix, annot=True, fmt=\".0f\", cmap =\"YlGnBu\")\r\n",
        "  plt.xlabel('REAL')\r\n",
        "  plt.ylabel('PRED')\r\n",
        "  plt.title('CONFUSION MATRIX')\r\n",
        "\r\n",
        "  # Colocar os nomes\r\n",
        "  ax.xaxis.set_ticklabels(['OUT', 'IN']) \r\n",
        "  ax.yaxis.set_ticklabels(['OUT', 'IN'])\r\n",
        "\r\n",
        "evaluate_SVM(X_test, y_test)"
      ],
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "# Train and predict SVM model with hyper-parameters tuning (GridSearchCV)\r\n",
        "\r\n",
        "\r\n",
        "tuned_parameters = [{'kernel': ['rbf', 'poly', 'linear'],\r\n",
        "                    'gamma': [100, 10, 1, 1e-1, 1e-2, 1e-3, 1e-4],\r\n",
        "                    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}]\r\n",
        "                     \r\n",
        "print(\"Hyperparameter Tuning for accuracy\")\r\n",
        "print()\r\n",
        "\r\n",
        "model = GridSearchCV(SVC(random_state=100), tuned_parameters, scoring='accuracy')\r\n",
        "model.fit(X_train, y_train)\r\n",
        "\r\n",
        "y_pred_SVM_GS = model.predict(X_test)\r\n",
        "print(classification_report(y_test, y_pred_SVM_GS))\r\n",
        "print()\r\n",
        "print(\"BEST TUNED PARAMETERS\", model.best_params_)"
      ],
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "# Evaluate SVM model with hyper-parameters tuning (GridSearchCV)\r\n",
        "\r\n",
        "def evaluate_SVM_GS(y_test, y_pred_SVM_GS):\r\n",
        "\r\n",
        "  # Acurácia\r\n",
        "  from sklearn.metrics import accuracy_score\r\n",
        "  accuracy = accuracy_score(y_test, y_pred_SVM_GS)\r\n",
        "  print('Acurácia: ', accuracy)\r\n",
        "\r\n",
        "  # Kappa\r\n",
        "  from sklearn.metrics import cohen_kappa_score\r\n",
        "  kappa = cohen_kappa_score(y_test, y_pred_SVM_GS)\r\n",
        "  print('Kappa: ', kappa)\r\n",
        "\r\n",
        "  # F1\r\n",
        "  from sklearn.metrics import f1_score\r\n",
        "  f1 = f1_score(y_test, y_pred_SVM_GS)\r\n",
        "  print('F1: ', f1)\r\n",
        "\r\n",
        "  # Matriz de confusão\r\n",
        "  from sklearn.metrics import confusion_matrix\r\n",
        "  confMatrix = confusion_matrix(y_pred_SVM_GS, y_test)\r\n",
        "\r\n",
        "  plt.figure(figsize=(20,12))\r\n",
        "  ax = plt.subplot()\r\n",
        "  sns.heatmap(confMatrix, annot=True, fmt=\".0f\", cmap =\"YlGnBu\")\r\n",
        "  plt.xlabel('REAL')\r\n",
        "  plt.ylabel('PRED')\r\n",
        "  plt.title('CONFUSION MATRIX')\r\n",
        "\r\n",
        "  # Colocar os nomes\r\n",
        "  ax.xaxis.set_ticklabels(['OUT', 'IN']) \r\n",
        "  ax.yaxis.set_ticklabels(['OUT', 'IN'])\r\n",
        "\r\n",
        "evaluate_SVM_GS(y_test, y_pred_SVM_GS)"
      ],
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": [
        "#Decision Tree"
      ],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "# Train and predict Decision Tree model\r\n",
        "\r\n",
        "def train_DT(X_train, y_train):\r\n",
        "  model = DecisionTreeClassifier(max_depth = 8, min_samples_leaf = 5, random_state=100)\r\n",
        "  model.fit(X_train, y_train);\r\n",
        "  return model\r\n",
        "\r\n",
        "model_DT = train_DT(X_train, y_train)\r\n",
        "y_pred_DT = model_DT.predict(X_test)"
      ],
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "# Plot decision tree\r\n",
        "\r\n",
        "plt.figure(figsize=(20, 10))\r\n",
        "tree.plot_tree(model_DT, class_names=[\"OUT\", \"IN\"], filled=True, rounded=True);"
      ],
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "# Train and predict Decision Tree model with hyper-parameters tuning (GridSearchCV)\r\n",
        "\r\n",
        "tuned_parameters = [{'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], \r\n",
        "                        'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]}]\r\n",
        "\r\n",
        "print(\"Hyperparameter Tuning for accuracy\")\r\n",
        "print()\r\n",
        "\r\n",
        "model = GridSearchCV(DecisionTreeClassifier(random_state=100), tuned_parameters, scoring='accuracy')\r\n",
        "model.fit(X_train, y_train)\r\n",
        "\r\n",
        "y_pred_DT_GS = model.predict(X_test)\r\n",
        "print(classification_report(y_test, y_pred_DT))\r\n",
        "print()\r\n",
        "print(\"BEST TUNED PARAMETERS\", model.best_params_)"
      ],
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "# Evaluate Decision Tree model with hyper-parameters tuning (GridSearchCV)\r\n",
        "\r\n",
        "def evaluate_DT_GS(y_test, y_pred_DT):  \r\n",
        "\r\n",
        "  # Acurácia\r\n",
        "  from sklearn.metrics import accuracy_score\r\n",
        "  accuracy = accuracy_score(y_test, y_pred_DT)\r\n",
        "  print('Acurácia: ', accuracy)\r\n",
        "\r\n",
        "  # Kappa\r\n",
        "  from sklearn.metrics import cohen_kappa_score\r\n",
        "  kappa = cohen_kappa_score(y_test, y_pred_DT)\r\n",
        "  print('Kappa: ', kappa)\r\n",
        "\r\n",
        "  # F1\r\n",
        "  from sklearn.metrics import f1_score\r\n",
        "  f1 = f1_score(y_test, y_pred_DT, average='weighted')\r\n",
        "  print('F1: ', f1)\r\n",
        "\r\n",
        "  # Matriz de confusão\r\n",
        "  from sklearn.metrics import confusion_matrix\r\n",
        "  confMatrix = confusion_matrix(y_pred_DT, y_test)\r\n",
        "\r\n",
        "  plt.figure(figsize=(20,12))\r\n",
        "  ax = plt.subplot()\r\n",
        "  sns.heatmap(confMatrix, annot=True, fmt=\".0f\", cmap =\"YlGnBu\")\r\n",
        "  plt.xlabel('REAL')\r\n",
        "  plt.ylabel('PRED')\r\n",
        "  plt.title('CONFUSION MATRIX')\r\n",
        "\r\n",
        "  # Colocar os nomes\r\n",
        "  ax.xaxis.set_ticklabels(['OUT', 'IN']) \r\n",
        "  ax.yaxis.set_ticklabels(['OUT', 'IN'])\r\n",
        "  plt.show()\r\n",
        "\r\n",
        "evaluate_DT_GS(y_test, y_pred_DT)"
      ],
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": [
        "#Random Forest"
      ],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "# Train and predict Random Forest model\r\n",
        "\r\n",
        "def train_RF(X_train, y_train):\r\n",
        "  model = RandomForestClassifier(min_samples_leaf=5, random_state=100)\r\n",
        "  model.fit(X_train, y_train);\r\n",
        "  return model\r\n",
        "\r\n",
        "model_RF = train_RF(X_train, y_train)\r\n",
        "y_pred_RF = model_RF.predict(X_test)"
      ],
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "model_RF.feature_importances_"
      ],
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "# Evaluate Random Forest model\r\n",
        "\r\n",
        "def evaluate_RF(y_test, y_pred_RF):  \r\n",
        "\r\n",
        "  # Acurácia\r\n",
        "  from sklearn.metrics import accuracy_score\r\n",
        "  accuracy = accuracy_score(y_test, y_pred_RF)\r\n",
        "  print('Acurácia: ', accuracy)\r\n",
        "\r\n",
        "  # Kappa\r\n",
        "  from sklearn.metrics import cohen_kappa_score\r\n",
        "  kappa = cohen_kappa_score(y_test, y_pred_RF)\r\n",
        "  print('Kappa: ', kappa)\r\n",
        "\r\n",
        "  # F1\r\n",
        "  from sklearn.metrics import f1_score\r\n",
        "  f1 = f1_score(y_test, y_pred_RF, average='weighted')\r\n",
        "  print('F1: ', f1)\r\n",
        "\r\n",
        "  # Matriz de confusão\r\n",
        "  from sklearn.metrics import confusion_matrix\r\n",
        "  confMatrix = confusion_matrix(y_pred_RF, y_test)\r\n",
        "\r\n",
        "  plt.figure(figsize=(20,12))\r\n",
        "  ax = plt.subplot()\r\n",
        "  sns.heatmap(confMatrix, annot=True, fmt=\".0f\", cmap=\"YlGnBu\")\r\n",
        "  plt.xlabel('REAL')\r\n",
        "  plt.ylabel('PRED')\r\n",
        "  plt.title('CONFUSION MATRIX')\r\n",
        "\r\n",
        "  # Colocar os nomes\r\n",
        "  ax.xaxis.set_ticklabels(['OUT', 'IN']) \r\n",
        "  ax.yaxis.set_ticklabels(['OUT', 'IN'])\r\n",
        "  plt.show()\r\n",
        "\r\n",
        "evaluate_RF(y_test, y_pred_RF)"
      ],
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "# Plot the importance of the attributes to the model\r\n",
        "\r\n",
        "importances = model_RF.feature_importances_\r\n",
        "std = np.std([tree.feature_importances_ for tree in model_RF.estimators_],\r\n",
        "             axis=0)\r\n",
        "indices = np.argsort(importances)\r\n",
        "\r\n",
        "# Plot the feature importances of the forest\r\n",
        "plt.figure(figsize=(20,50))\r\n",
        "plt.title(\"Feature importances\")\r\n",
        "plt.barh(range(X_train.shape[1]), importances[indices],\r\n",
        "       color=\"g\", align=\"center\")\r\n",
        "# If you want to define your own labels,\r\n",
        "# change indices to a list of labels on the following line.\r\n",
        "plt.yticks(range(X_train.shape[1]), indices)\r\n",
        "plt.ylim([-1, X_train.shape[1]])\r\n",
        "plt.show()"
      ],
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "# Train and predict Random Forest model with hyper-parameters tuning (GridSearchCV)\r\n",
        "\r\n",
        "tuned_parameters = [{'n_estimators': [20, 50, 100, 150, 200, 300, 400, 500],\r\n",
        "                     'max_features': [3,4,8,9,10,11]}]\r\n",
        "\r\n",
        "print(\"Hyperparameter Tuning for accuracy\")\r\n",
        "print()\r\n",
        "\r\n",
        "model = GridSearchCV(RandomForestClassifier(n_jobs=50, verbose=1, random_state=100), tuned_parameters, scoring='accuracy')\r\n",
        "model.fit(X_train, y_train)\r\n",
        "\r\n",
        "y_pred_RF = model.predict(X_test)\r\n",
        "print(classification_report(y_pred, y_test))\r\n",
        "print()\r\n",
        "print(\"BEST TUNED PARAMETERS\", model_XGB_GS.best_params_)"
      ],
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "# Evaluate Random Forest model with hyper-parameters tuning (GridSearchCV)\r\n",
        "\r\n",
        "def evaluate_RF_GS(y_test, y_pred_RF):  \r\n",
        "\r\n",
        "  # Acurácia\r\n",
        "  from sklearn.metrics import accuracy_score\r\n",
        "  accuracy = accuracy_score(y_test, y_pred_RF)\r\n",
        "  print('Acurácia: ', accuracy)\r\n",
        "\r\n",
        "  # Kappa\r\n",
        "  from sklearn.metrics import cohen_kappa_score\r\n",
        "  kappa = cohen_kappa_score(y_test, y_pred_RF)\r\n",
        "  print('Kappa: ', kappa)\r\n",
        "\r\n",
        "  # F1\r\n",
        "  from sklearn.metrics import f1_score\r\n",
        "  f1 = f1_score(y_test, y_pred_RF, average='weighted')\r\n",
        "  print('F1: ', f1)\r\n",
        "\r\n",
        "  # Matriz de confusão\r\n",
        "  from sklearn.metrics import confusion_matrix\r\n",
        "  confMatrix = confusion_matrix(y_pred_RF, y_test)\r\n",
        "\r\n",
        "  plt.figure(figsize=(20,12))\r\n",
        "  ax = plt.subplot()\r\n",
        "  sns.heatmap(confMatrix, annot=True, fmt=\".0f\", cmap=\"YlGnBu\")\r\n",
        "  plt.xlabel('REAL')\r\n",
        "  plt.ylabel('PRED')\r\n",
        "  plt.title('CONFUSION MATRIX')\r\n",
        "\r\n",
        "  # Colocar os nomes\r\n",
        "  ax.xaxis.set_ticklabels(['OUT', 'IN']) \r\n",
        "  ax.yaxis.set_ticklabels(['OUT', 'IN'])\r\n",
        "  plt.show()\r\n",
        "\r\n",
        "evaluate_RF_GS(y_test, y_pred_RF)"
      ],
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": [
        "#Gradient Boosting Classifier"
      ],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "# Train and predict Gradient Boosting Classifier\r\n",
        "\r\n",
        "def train_GBC (X_train, y_train):\r\n",
        "    model = GradientBoostingClassifier(n_estimators=1,\r\n",
        "                                        min_samples_leaf=5,\r\n",
        "                                        learning_rate=0.1,\r\n",
        "                                        loss='deviance',\r\n",
        "                                        subsample=1.0,\r\n",
        "                                        criterion='fiedman_mse',\r\n",
        "                                        min_samples_split=2,\r\n",
        "                                        max_depth=8,\r\n",
        "                                        random_state=100\r\n",
        "                                        )\r\n",
        "    model.fit(X_train, y_train)\r\n",
        "    return model\r\n",
        "\r\n",
        "model_GBC = train_GBC(X_train, y_train)\r\n",
        "y_pred_GBC = model_GBC.predict(X_test)"
      ],
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "# Evaluate Gradient Boosting Classifier model\r\n",
        "\r\n",
        "def evaluate_GBC(y_test, y_pred_GBC):\r\n",
        "\r\n",
        "  # Acurácia\r\n",
        "  from sklearn.metrics import accuracy_score\r\n",
        "  accuracy = accuracy_score(y_test, y_pred_GBC)\r\n",
        "  print('Acurácia: ', accuracy)\r\n",
        "\r\n",
        "  # Kappa\r\n",
        "  from sklearn.metrics import cohen_kappa_score\r\n",
        "  kappa = cohen_kappa_score(y_test, y_pred_GBC)\r\n",
        "  print('Kappa: ', kappa)\r\n",
        "\r\n",
        "  # F1\r\n",
        "  from sklearn.metrics import f1_score\r\n",
        "  f1 = f1_score(y_test, y_pred_GBC)\r\n",
        "  print('F1: ', f1)\r\n",
        "\r\n",
        "  # Matriz de confusão\r\n",
        "  from sklearn.metrics import confusion_matrix\r\n",
        "  confMatrix = confusion_matrix(y_pred_GBC, y_test)\r\n",
        "\r\n",
        "  plt.figure(figsize=(20,12))\r\n",
        "  ax = plt.subplot()\r\n",
        "  sns.heatmap(confMatrix, annot=True, fmt=\".0f\", cmap =\"YlGnBu\")\r\n",
        "  plt.xlabel('REAL')\r\n",
        "  plt.ylabel('PRED')\r\n",
        "  plt.title('CONFUSION MATRIX')\r\n",
        "\r\n",
        "  # Colocar os nomes\r\n",
        "  ax.xaxis.set_ticklabels(['OUT', 'IN']) \r\n",
        "  ax.yaxis.set_ticklabels(['OUT', 'IN'])\r\n",
        "\r\n",
        "evaluate_GBC(X_test, y_test)"
      ],
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": [
        "#XGBOOST"
      ],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "# Train and predict XGBOOST model\r\n",
        "\r\n",
        "def train_XGB (X_train, y_train):\r\n",
        "    model = xgb.XGBClassifier(n_estimators=2000, \r\n",
        "                              objective=\"reg:logistic\", \r\n",
        "                              booster='gbtree', \r\n",
        "                              alpha=10, \r\n",
        "                              max_depth=8,\r\n",
        "                              gamma=1e-2,\r\n",
        "                              colsample_bytree=0.8,\r\n",
        "                              random_state=100)\r\n",
        "    model.fit(X_train, y_train)\r\n",
        "    return model\r\n",
        "\r\n",
        "model_XGB = train_XGB(X_train, y_train)\r\n",
        "y_pred_XGB = model_XGB.predict(X_test)"
      ],
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "# Evaluate XGBOOST model\r\n",
        "\r\n",
        "def evaluate_XGB(X_test, y_test):\r\n",
        "\r\n",
        "  # Acurácia\r\n",
        "  from sklearn.metrics import accuracy_score\r\n",
        "  accuracy = accuracy_score(y_test, y_pred_XGB)\r\n",
        "  print('Acurácia: ', accuracy)\r\n",
        "\r\n",
        "  # Kappa\r\n",
        "  from sklearn.metrics import cohen_kappa_score\r\n",
        "  kappa = cohen_kappa_score(y_test, y_pred_XGB)\r\n",
        "  print('Kappa: ', kappa)\r\n",
        "\r\n",
        "  # F1\r\n",
        "  from sklearn.metrics import f1_score\r\n",
        "  f1 = f1_score(y_test, y_pred_XGB)\r\n",
        "  print('F1: ', f1)\r\n",
        "\r\n",
        "  # Matriz de confusão\r\n",
        "  from sklearn.metrics import confusion_matrix\r\n",
        "  confMatrix = confusion_matrix(y_pred_XGB, y_test)\r\n",
        "\r\n",
        "  plt.figure(figsize=(20,12))\r\n",
        "  ax = plt.subplot()\r\n",
        "  sns.heatmap(confMatrix, annot=True, fmt=\".0f\", cmap =\"YlGnBu\")\r\n",
        "  plt.xlabel('REAL')\r\n",
        "  plt.ylabel('PRED')\r\n",
        "  plt.title('CONFUSION MATRIX')\r\n",
        "\r\n",
        "  # Colocar os nomes\r\n",
        "  ax.xaxis.set_ticklabels(['OUT', 'IN']) \r\n",
        "  ax.yaxis.set_ticklabels(['OUT', 'IN'])\r\n",
        "\r\n",
        "evaluate_XGB(X_test, y_test)"
      ],
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "# Plot the importance of the attributes to the model\r\n",
        "\r\n",
        "xgb.plot_importance(model_XGB)\r\n",
        "plt.rcParams['figure.figsize'] = [5, 5]\r\n",
        "plt.show()\r\n"
      ],
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "# Train and predict XGBOOST model with hyper-parameters tuning (GridSearchCV)\r\n",
        "\r\n",
        "tuned_parameters = {\r\n",
        "                    #'learning_rate': [0.1, 0.3, 0.5, 0.7, 0.9],\r\n",
        "                    #'max_depth': [8,9,10,11,12,13], \r\n",
        "                    #'alpha': [5,6,7,8,9,10,11,12,13,14,15,17,18,19,20],\r\n",
        "                    #'lambda': [5,6,7,8,9,10,11,12,13,14,15,17,18,19,20]\r\n",
        "                    #'min_child_weight': [1, 5, 10],\r\n",
        "                    #'colsample_bytree': [0.6, 0.8, 1.0]\r\n",
        "                    #'gamma': [1e-1, 1e-2, 1e-3, 1e-4],\r\n",
        "                    #'n_estimators': [50, 100, 200, 300, 400, 500, 1000]\r\n",
        "                    }\r\n",
        "\r\n",
        "print('Hyperparameter Tuning for accuracy')\r\n",
        "\r\n",
        "model_XGB_GS = GridSearchCV(xgb.XGBClassifier(\r\n",
        "                                            objective='reg:squarederror',\r\n",
        "                                            n_estimators=100,\r\n",
        "                                            use_label_encoder=False,\r\n",
        "                                            random_state=100,\r\n",
        "                                            alpha=10,\r\n",
        "                                            max_depth=8,\r\n",
        "                                            gamma=1e-3,\r\n",
        "                                            learning_rate=0.3,\r\n",
        "                                            min_child_weight=0.0001,\r\n",
        "                                            #lambda=7,\r\n",
        "                                            ), \r\n",
        "                                              tuned_parameters, )\r\n",
        "model_XGB_GS.fit(X_train, y_train)\r\n",
        "\r\n",
        "y_pred_XGB_GS = model_XGB_GS.predict(X_test)\r\n",
        "print(classification_report(y_test, y_pred_XGB_GS))\r\n",
        "print()\r\n",
        "print(\"BEST TUNED PARAMETERS\", model_XGB_GS.best_params_)\r\n",
        "                    "
      ],
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "# Evaluate XGBOOST model with hyper-parameters tuning (GridSearchCV)\r\n",
        "\r\n",
        "def evaluate_XGB_CV(y_test, y_pred_XGB_GS):\r\n",
        "\r\n",
        "  # Acurácia\r\n",
        "  from sklearn.metrics import accuracy_score\r\n",
        "  accuracy = accuracy_score(y_test, y_pred_XGB_GS)\r\n",
        "  print('Acurácia: ', accuracy)\r\n",
        "\r\n",
        "  # Kappa\r\n",
        "  from sklearn.metrics import cohen_kappa_score\r\n",
        "  kappa = cohen_kappa_score(y_test, y_pred_XGB_GS)\r\n",
        "  print('Kappa: ', kappa)\r\n",
        "\r\n",
        "  # F1\r\n",
        "  from sklearn.metrics import f1_score\r\n",
        "  f1 = f1_score(y_test, y_pred_XGB_GS)\r\n",
        "  print('F1: ', f1)\r\n",
        "\r\n",
        "  # Matriz de confusão\r\n",
        "  from sklearn.metrics import confusion_matrix\r\n",
        "  confMatrix = confusion_matrix(y_pred_XGB_GS, y_test)\r\n",
        "\r\n",
        "  plt.figure(figsize=(20,12))\r\n",
        "  ax = plt.subplot()\r\n",
        "  sns.heatmap(confMatrix, annot=True, fmt=\".0f\", cmap =\"YlGnBu\")\r\n",
        "  plt.xlabel('REAL')\r\n",
        "  plt.ylabel('PRED')\r\n",
        "  plt.title('CONFUSION MATRIX')\r\n",
        "\r\n",
        "  # Colocar os nomes\r\n",
        "  ax.xaxis.set_ticklabels(['OUT', 'IN']) \r\n",
        "  ax.yaxis.set_ticklabels(['OUT', 'IN'])\r\n",
        "\r\n",
        "evaluate_XGB_CV(y_test, y_pred_XGB_GS)"
      ],
      "outputs": [],
      "metadata": {}
    }
  ]
}
