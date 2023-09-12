# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 15:37:00 2023

@author: gaspr
"""

import pandas as pd 
import seaborn as sns 
import streamlit as st 
import matplotlib.pyplot as plt 
import plotly.express as px
import numpy as np 
import calendar
import geopandas as gpd
import folium
#import joblib
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier
#from sklearn import ensemble, model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
#from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, mean_squared_error, mean_absolute_error, explained_variance_score



df=pd.read_csv('df_new.csv')
df_inc=pd.read_csv("df_inc1.csv")
df_mob=pd.read_csv("df_mob1.csv")
data=pd.read_csv('data.csv')
df_inc_dist=pd.read_csv("df_inc_dist.csv")
df_mob_dist=pd.read_csv("df_mob_dist.csv")
df_coordo=pd.read_csv("df_coordo.csv")
df_dista=pd.read_csv("df_dista.csv")
df_dist=pd.read_csv("df_dist.csv")



st.sidebar.markdown("<h1 style='font-family: lato;font-size:34px; color:#B22222;text-align: center'>Réactivité de la Brigade des Pompiers de Londres</h1>", unsafe_allow_html=True)


pages=["Introduction","Datasets","Exploration des données","Visualisation des données","Modélisation des données","Calcul des distances et application dummy"]

page=st.sidebar.radio("Sélection :", pages)
st.sidebar.write("\n")
#st.sidebar.write("\n")

st.sidebar.write("Auteurs :")
st.sidebar.write("AMBROISE Gladimir")
st.sidebar.write("RODRIGUEZ Rosa")
st.sidebar.write("HAMMOUDA Elmahdy")
st.sidebar.write("\n")
#st.sidebar.write("\n")

st.sidebar.write("Promotion juillet 2022 : Formation continue")

st.sidebar.markdown("<h1 style='text-align: center; font-size:30px; color:#87CEEB;'>DataScientest</h1>", unsafe_allow_html=True)
st.sidebar.write("\n")

st.sidebar.write("Sources :")
st.sidebar.write("https://www.london-fire.gov.uk/")
st.sidebar.write("https://data.london.gov.uk/")

    
    
st.markdown("<h1 style='text-align: center; font-size: 42px;'>Py Rescue Team</h1>", unsafe_allow_html=True)

import base64
from pathlib import Path

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded
def img_to_html(img_path):
    img_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
      img_to_bytes(img_path)
    )
    return img_html

st.markdown("<p style='text-align: center; color: grey;'>"+img_to_html('LFB Mod.jpg')+"</p>", unsafe_allow_html=True)

if page==pages[0]:
    st.markdown("<h1 style='text-align: center; font-size: 25px;'>'Serve and protect'</h1>", unsafe_allow_html=True)
    
    st.write("""La :red[London Fire Brigade] est dans le top 5 des plus grands corps de sapeurs-pompiers du monde.""")
    st.write("""Avec une zone de couverture de plus de 1 587 mètres carrés et une moyenne de 100 000 incidents par an, leur réactivité doit 
             être optimisée.""")
             
    st.write("""Dans le cadre de notre projet, nous allons analyser le temps de réponse et de mobilisation de la Brigade des Pompiers de Londres vers différents types d’incidents. 
             Ensuite, nous développerons un modèle capable de fournir à l’appelant une estimation de ce temps de réponse.""")
    
    st.write("""Pour commencer, nous allons explorer et nettoyer les données présentes dans les deux datasets fournis.""")
    
    st.write("""Après une visualisation de ces données, nous procéderons à leur modélisation afin de répondre à notre problématique.""")


if page==pages[1]:   #Datasets
    st.markdown("<h1 style='text-align: center; font-size: 25px;'>Datasets</h1>", unsafe_allow_html=True)
    st.write("""Pour la réalisation de ce projet on a eu accès à deux bases de données.""")
             
    st.write("""La première est nommée- LFB Incident data - last three years- et contient les détails de chaque incident traité par la LFB depuis janvier 2009. 
             Dans les lignes on trouve des informations concernant la date, le lieu de l'incident ainsi que le type d'incident traité.""")
    
    lien1="https://data.london.gov.uk/dataset/london-fire-brigade-incident-records"
    texte_du_lien="https://data.london.gov.uk/dataset/london-fire-brigade-incident-records "
    st.markdown(f"[{texte_du_lien}]({lien1})")         
    
    st.write("""La seconde base de données est nommée- LFB Mobilisation data - last three years. Elle contient les détails de chaque camion de pompiers envoyé 
             sur les lieux d'un incident depuis janvier 2009. On peut trouver des informations comme l'appareil mobilisé, son lieu de déploiement et les heures 
             d'arrivée sur les lieux de l'incident.""")             
     
    lien2="https://data.london.gov.uk/dataset/london-fire-brigade-mobilisation-records"
    texte_du_lien="https://data.london.gov.uk/dataset/london-fire-brigade-mobilisation-records"
    st.markdown(f"[{texte_du_lien}]({lien2})")  
    
    st.write("""Ces deux bases de données sont accessibles depuis le portail ‘London DataStore’ qui héberge de façon gratuite de la data concernant la capitale londonienne.""")
    
    
if page==pages[2]:  #Exploration
    st.markdown("<h1 style='text-align: center; font-size: 25px;'>Exploration</h1>", unsafe_allow_html=True)
    
    st.write("<ul><li><span style='font-size: 20px;'>LFB Incident data Last 3 years</span></li></ul>", unsafe_allow_html=True)

    
    line_to_plot=st.slider("Sélectionner nombre de lignes à afficher:",key=1,min_value=3,max_value=100)
    st.dataframe(df_inc.sample(line_to_plot))
    
    st.write("\n")
    st.write("\n")
    
    st.write("<ul><li><span style='font-size: 20px;'>LFB Mobilisation data Last 3 years</span></li></ul>", unsafe_allow_html=True)

    
    line_to_plot1=st.slider("Sélectionner nombre de lignes à afficher:",key=2,min_value=3,max_value=100)
    st.dataframe(df_mob.head(line_to_plot1))
    
    st.write("\n")
    st.write("\n")
    
    st.info("Grâce à la variable 'IncidentNumber' présente dans les deux bases de données nous avons procédé à leur fusion.")
    
    st.write("<ul><li><span style='font-size: 20px;'>Base de données prête pour le modèle</span></li></ul>", unsafe_allow_html=True)

    line_to_plot2=st.slider("Sélectionner nombre de lignes à afficher:",key=4,min_value=3,max_value=100)
    st.dataframe(df.head(line_to_plot2))

    #st.dataframe(df.rename(columns = {'DateOfCall':'Dateappel',
    #                      'TurnoutTimeSeconds':'Tempspréparation',
    #                      'IncidentStationGround':'StationProcheIncident',
    #                      'year':'année',
    #                      'PropertyCategory':'Typedepropriete',
    #                      'ProperCase':'NomArrondissement',
    #                      'month':'mois',
    #                      'AttendanceTimeSeconds': 'TempsArrivée',
    #                      'DateAndTimeMobilised':'DateETempsMobilisation'
    #                      }))

    st.write("\n")
    st.write("\n")
    if st.checkbox("Aficher les valeurs manquantes"):
        st.dataframe(df_inc.isna().sum())
    
    st.write("Description statistique de la base de données:")
    st.dataframe(df.describe())

    #st.markdown( "Estimation du temps de réponse et mobilisation de la [Brigade des Pompiers de Londres](https://www.london-fire.gov.uk/)")

        
if page==pages[3]:  #Visualisation

    st.markdown("<h1 style='text-align: center; font-size: 25px;'>Visualisation</h1>", unsafe_allow_html=True)

    st.markdown("<p style='font-size: 15px;'>Dans la partie Data'Viz, nous avons utilisé les deux bases df_inc et df_mob nettoyées, ainsi qui une autre nommée data.</p>", unsafe_allow_html=True)
    
    st.write("<ul><li><span style='font-size: 20px;'>Base de données pour la visualisation: data</span></li></ul>", unsafe_allow_html=True)

    if st.checkbox("Afficher la base de données:",key=3):
        line_to_plot2=st.slider("Sélectionner nombre de lignes à afficher:",key=4,min_value=3,max_value=100)
        st.dataframe(data.head(line_to_plot2))

    st.write("<ul><li><span style='font-size: 20px;'>Graphes :</span></li></ul>", unsafe_allow_html=True)

    
    st.markdown("<h1 style='text-align: center; font-size: 18px;'>Nombre d'incidents par année</h1>", unsafe_allow_html=True)

    fig=plt.figure()
    incident_counts = df_inc.groupby('CalYear')['IncidentNumber'].count()
    incident_counts = incident_counts[incident_counts.index <= 2021]
    plt.plot(incident_counts.index, incident_counts.values, marker='o',color='red')
    plt.xlabel('Année')
    plt.ylabel("Nombre d'incidents")
    plt.title("Nombre d'incidents par année")
    plt.xticks(incident_counts.index.astype(int))
    st.pyplot(fig)
    
    st.markdown("<p style='font-size: 15px;'>Dans ce premier graphique, nous observons l'évolution du nombre d'incidents entre 2019 et 2021, les années pour lesquels on a des données de janvier à" 
                " décembre. Si de 2019 à 2020 nous avons une diminution d’environ 6% des appels reçus, 2021 enregistre une augmentation de 11%.</p>", unsafe_allow_html=True)

    ###
    st.markdown("<h1 style='text-align: center; font-size: 18px;'>Temps moyen de trajet par année</h1>", unsafe_allow_html=True)

    fig=plt.figure()
    df_mob['CalYear'] = pd.to_datetime(df_mob['CalYear'], format='%Y').dt.year

    mean_travel_time_by_year = df_mob.groupby('CalYear')['TravelTimeSeconds'].mean()

    fig, ax = plt.subplots()
    mean_travel_time_by_year = mean_travel_time_by_year[mean_travel_time_by_year.index <= 2021]
    ax.plot(mean_travel_time_by_year.index.astype(str),((mean_travel_time_by_year.values)//60)+(mean_travel_time_by_year.values%60/100), marker='o',color='red')
    ax.set_xlabel('Année')
    ax.set_ylabel('Temps moyen de trajet (minutes)')
    st.pyplot(fig)
    
    st.markdown("<p style='font-size: 15px;'>Quand nous analysons le temps de trajet moyen, nous remarquons que si leur temps de déplacement est similaire pour les années 2019 et 2021, en 2020,"
                " la LFB arrive en moyenne 10 seconds plus rapidement au lieu de l’incident qu’en 2019.</p>", unsafe_allow_html=True)


    ###
    st.markdown("<h1 style='text-align: center; font-size: 18px;'>Histogramme du temps de réponse pour arriver sur place</h1>", unsafe_allow_html=True)

    fig,ax=plt.subplots()                                                                        

    ax.hist(data['AttendanceTimeSeconds'],bins=50,color='red')
    plt.xlabel('Temps de réponse (seconds)')
    plt.ylabel('Nombre d\'appels')
    st.pyplot(fig)

    
    st.markdown("<p style='font-size: 15px;'>L'histogramme du temps de réponse pour arriver sur place indique que la majorité des interventions ont un temps de réponse compris entre 200 et 500 secondes" 
                " (0:3:20-0:8:20). Malgré le nombre élevé d'appels (supérieur à 20 000), cela suggère une réponse relativement rapide dans l'ensemble. La distribution des temps de réponse reste concentrée"
                " dans cette plage,ce qui est encourageant en termes de rapidité d'intervention. Nous soulignons également que le temps de réponse peut être considéré comme rapide pour toutes les"
                " interventions confondues.</p>", unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center; font-size: 18px;'>Noms des différentes catégories présentes dans la colonne 'PropertyCategory'</h1>", unsafe_allow_html=True)

    st.write("Catégories différentes :")
    st.dataframe(data['PropertyCategory'].unique())
    
    ###
    st.markdown("<h1 style='text-align: center; font-size: 18px;'>Boxplot du temps de réponse en fonction de la catégorie de propriété</h1>", unsafe_allow_html=True)

    fig=plt.figure()
    atime=data['AttendanceTimeSeconds']
    sns.boxplot(x='PropertyCategory', y=(atime.values//60)+(atime.values%60/100), data = data)
    plt.xlabel('Catégorie de propriété')
    plt.ylabel('Temps de réponse (minutes)')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    
    st.markdown("<p style='font-size: 15px;'>Cette visualisation permet de comparer la distribution du temps d'arrivée en fonction de la catégorie de propriété. Nous pouvons observer que la plupart"
                " des catégories de propriétés ont des temps d'arrivée similaires, ce qui suggère un traitement dans le même ordre de grandeur. Cependant, les catégories 'Aircraft' (aéronef) et"
                 " 'Boat' (bateau) se distinguent avec des temps d'intervention plus longs. On peut donc dire que pour les catégories hors sol, le temps d'intervention est plus grand, ce qui semble"
                " logique au vu des ressources spécifiques à mettre en place pour accéder au lieu de l'incident</p>", unsafe_allow_html=True)
    

    ###
    #st.write("**Diagramme en barres du nombre d'appels par mois**")
    #st.markdown("<h1 style='text-align: center; font-size: 18px;'>Diagramme en barres du nombre d'appels par mois</h1>", unsafe_allow_html=True)

    # fig=plt.figure()
    # calls_by_month = data['month'].value_counts()
    # calls_by_month = calls_by_month.sort_index()
    # months = [calendar.month_name[i] for i in range(1, 13)]
    # plt.bar(calls_by_month.index, calls_by_month.values,color='red')
    # plt.xticks(calls_by_month.index, months,rotation=45)
    # plt.xlabel('Mois')
    # plt.ylabel('Nombre d\'appels')
    # st.pyplot(fig)


    ####Graphe interactif:

    st.markdown("<h1 style='text-align: center; font-size: 18px;'>Diagramme en barres du nombre d'appels par mois</h1>", unsafe_allow_html=True)

    calls_by_month = data['month'].value_counts()
    calls_by_month = calls_by_month.sort_index()
    months = [calendar.month_name[i] for i in range(1, 13)]
    fig=px.bar(calls_by_month, x=calls_by_month.index,y=calls_by_month.values,labels={'x':'Mois','y':"Nombre d'appels"},color_discrete_sequence=['red'])
    fig.update_xaxes(title='Mois',showticklabels=True, tickangle=0, tickmode='array', ticktext=months, tickvals=calls_by_month.index)
    fig.update_yaxes(showticklabels=True)
    fig.update_layout(xaxis={'tickmode': 'array', 'ticktext': months, 'tickvals': calls_by_month.index, 'position': 0.0})
    st.plotly_chart(fig)
    
    st.markdown("<p style='font-size: 15px;'>Cette visualisation nous permet de voir la distribution des appels au fil des mois. Nous pouvons observer une tendance saisonnière : en été, de juin à"
                " août, le nombre d'appel est au plus haut alors qu'en automne, de septembre à décembre, la brigade des pompiers de Londres est moins démandée.</p>", unsafe_allow_html=True)
    
    ###
    st.markdown("<h1 style='text-align: center; font-size: 18px;'>Temps de préparation moyen par station d'incendie et par mois</h1>", unsafe_allow_html=True)

    response_time_by_station = data.groupby(['IncidentStationGround'])['TurnoutTimeSeconds'].mean().reset_index()
    pivot_table = pd.pivot_table(data, values='TurnoutTimeSeconds', index=['IncidentStationGround'], columns=['month'], aggfunc=np.mean)
    fig=plt.figure(figsize=(16, 10))
    heatmap=sns.heatmap(pivot_table, cmap='coolwarm', linecolor='white', linewidths=1)
    plt.subplots_adjust(left=0.70)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0,fontsize=6)
    plt.yticks(np.arange(len(pivot_table.index)) + 0.5, pivot_table.index)
    plt.xlabel('Mois')
    plt.ylabel('Station d\'incendie')
    plt.xticks(ticks=np.arange(0.4, len(months) + 0.4), labels=months,rotation=65)
    st.pyplot(fig)
    
    
    st.markdown("<p style='font-size: 15px;'>Ce graphique permet de visualiser les temps de préparation moyens (en seconds) par station et par mois sous forme de carte de chaleur. Les tons de rouge indiquent des"
                " temps de réponse plus longs et les tons bleu indiquent des temps de réponse plus courts. On constate que certaines stations comme Biggin Hill, Orpington ou Bromley ont un temps"
                " de réponse important par rapport aux autres stations. De plus, le temps de réponse évolue sur l'année de la même manière pour chaque station, avec une"
                " période critique pour toutes en juin.</p>", unsafe_allow_html=True)

    ###
    st.markdown("<h1 style='text-align: center; font-size: 18px;'>Temps de réponse moyen par catégorie de propriété et par station d'incendie</h1>", unsafe_allow_html=True)

    # On regroupe par catégorie de propriété et station d'incendie et on calcule la moyenne des temps d'intervention
    prop_station = data.groupby(['PropertyCategory', 'IncidentStationGround'])[['AttendanceTimeSeconds']].mean().reset_index()

    # création d'une matrice pivot avec la moyenne des temps d'intervention pour chaque combinaison de propriété et de station
    pivot_table = prop_station.pivot('PropertyCategory', 'IncidentStationGround', 'AttendanceTimeSeconds')

    # Créer une carte de chaleur avec la moyenne des temps d'intervention pour chaque combinaison de propriété et de station
    fig=plt.figure(figsize=(16,8))
    sns.heatmap(pivot_table, cmap='coolwarm', annot=False, fmt='.0f', linewidths=.5)
    plt.xlabel('Station d\'incendie', fontsize=14)
    plt.ylabel('Catégorie de propriété', fontsize=14)
    st.pyplot(fig)
    
    st.markdown("<p style='font-size: 15px;'>Ce heatmap montre les temps d'intervention moyens pour chaque station d'incendie et catégorie de propriété.</p>"
            "<p style='font-size: 15px;'>Grâce à lui, on peut cibler les catégories de propriété qui posent problème par station d'incendie.</p>"
            "<p style='font-size: 15px;'>Par exemple, pour la station Biggin Hill, son temps de réponse est élevé pour toute catégorie sauf  'Outdoor structure'. On peut ainsi, pour un autre projet,"
            " développer une analyse en se concentrant seulement sur cette station et les interventions sur 'Outdoor structure'.</p>"
            "<p style='font-size: 15px;'>À l'aide d'une matrice de corrélation et en gardant toutes les données initiales de la base de données, nous pourrions identifier les facteurs qui ont une"
            " forte corrélation avec le temps de réponse pour les interventions à Biggin Hill sur les structures extérieures.</p>"
            "<p style='font-size: 15px;'>De manière générale, on constate grâce à ce graphique que, quelle que soit la station d'incendie, les interventions dans les logements (Dwelling), en"
            " extérieur (Outdoor) et sur les véhicules de transport (Transport Vehicle) ont les temps les plus médiocres.</p>", unsafe_allow_html=True)


    st.markdown("<h1 style='text-align: center; font-size: 18px;'>Géolocalisation</h1>", unsafe_allow_html=True)


    st.markdown("<p style='font-size: 15px;'>Afin de produire une carte représentant les stations d'incendies et leurs caractéristiques,nous décidons de nous penchez sur les"
                " variables 'Latitude' et 'Longitude'. Mais ces dernières présentent énormément de valeurs manquantes. Afin d'y remédier et de ne pas perdre plus de la moitié des données "
                " (Latitude et longitude ont respectivement près de 60% de NAN), nous décidons d'utiliser les variables IncGeo_BoroughName et IncGeo_WardName afin de créer artificiellement des"
                " coordonnées géographiques. Nous le réalisons en utilisant un dictionnaire de mapping pour associer les combinaisons de Borough et Ward aux coordonnées correspondantes.</h1>", unsafe_allow_html=True)

    ##Version Elm
    # coord_mapping = {}
    # grouped_data = data.groupby(['IncGeo_BoroughName', 'IncGeo_WardName'])
    # for (borough, ward), group in grouped_data:
    #     valid_coords = group[['Latitude', 'Longitude']].dropna()
    #     if not valid_coords.empty:
    #         coord_mapping[(borough, ward)] = valid_coords.iloc[0]

    # data[['Latitude', 'Longitude']] = data[['Latitude', 'Longitude']].fillna(data.apply(lambda row: coord_mapping.get((row['IncGeo_BoroughName'], row['IncGeo_WardName'])), axis=1))

    # sample_size = 1000  
    # data_sample = data.sample(n=sample_size, random_state=42)
    # data_sample['Longitude'] = data_sample['Longitude'].str.replace(',', '.')
    # data_sample['Latitude'] = data_sample['Latitude'].str.replace(',', '.')
    # gdf_incident = gpd.GeoDataFrame(data_sample, geometry=gpd.points_from_xy(data_sample['Longitude'], data_sample['Latitude']))
    
    # m = folium.Map(location=[51.5074, -0.1278], zoom_start=10)
    
    # for idx, row in gdf_incident.iterrows():
    #     tooltip = f"Nom : {row['ProperCase']}<br>" \
    #           f"Temps d'intervention : {row['AttendanceTimeSeconds']} secondes<br>" \
    #           f"Type de propriété : {row['PropertyCategory']}<br>" \
    #           f"Station : {row['IncidentStationGround']}<br>" \
    #           f"Arrondissement : {row['IncGeo_BoroughName']}<br>" \
    #           f"Quartier : {row['IncGeo_WardName']}"
    #     folium.Marker(location=[row['Latitude'], row['Longitude']], tooltip=tooltip).add_to(m)

    # st.write(m)

    ##Version avant modifs 
    # coord_mapping = {}
    # grouped_data = data.groupby(['IncGeo_BoroughName', 'IncGeo_WardName'])
    # for (borough, ward), group in grouped_data:
    #     valid_coords = group[['Latitude', 'Longitude']].dropna()
    #     if not valid_coords.empty:
    #         coord_mapping[(borough, ward)] = valid_coords.iloc[0]
    
    # data[['Latitude', 'Longitude']] = data[['Latitude', 'Longitude']].fillna(data.apply(lambda row: coord_mapping.get((row['IncGeo_BoroughName'], row['IncGeo_WardName'])), axis=1))
     
    # line_to_plot4=st.slider("Sélectionner nombre de stations à afficher:",key=1,min_value=1,max_value=50)
    # data_sample = data.sample(n=line_to_plot4, random_state=42)
    # data_sample['Longitude'] = data_sample['Longitude'].str.replace(',', '.')
    # data_sample['Latitude'] = data_sample['Latitude'].str.replace(',', '.')
    # gdf_incident = gpd.GeoDataFrame(data_sample, geometry=gpd.points_from_xy(data_sample['Longitude'].astype(float), data_sample['Latitude'].astype(float)))
    
    # m = folium.Map(location=[51.5074, -0.1278], zoom_start=10)
    # marker = folium.Marker(
    #     location=[51.50335, -0.09862],
    #     popup='London Fire Brigade',
    #     icon=folium.Icon(icon="fire",prefix="fa",color='red'))
    # marker.add_to(m)
    
   
    # for idx, row in gdf_incident.iterrows():
    #     attendance_time = row['AttendanceTimeSeconds']
    #     radius = attendance_time / 50
    #     tooltip = f"Nom : {row['ProperCase']}<br>" \
    #           f"Temps d'intervention : {row['AttendanceTimeSeconds']} secondes<br>" \
    #           f"Type de propriété : {row['PropertyCategory']}<br>" \
    #           f"Station : {row['IncidentStationGround']}<br>" \
    #           f"Arrondissement : {row['IncGeo_BoroughName']}<br>" \
    #           f"Quartier : {row['IncGeo_WardName']}"
    #     folium.CircleMarker(location=[row['Latitude'], row['Longitude']],radius=radius,color='red', tooltip=tooltip).add_to(m)
    
    # st_data = st_folium(m, width=725)

    coord_mapping = {}
    grouped_data = data.groupby(['IncGeo_BoroughName', 'IncGeo_WardName'])
    for (borough, ward), group in grouped_data:
        valid_coords = group[['Latitude', 'Longitude']].dropna()
        if not valid_coords.empty:
            coord_mapping[(borough, ward)] = valid_coords.iloc[0]
    
    data[['Latitude', 'Longitude']] = data[['Latitude', 'Longitude']].fillna(data.apply(lambda row: coord_mapping.get((row['IncGeo_BoroughName'], row['IncGeo_WardName'])), axis=1))
     
    line_to_plot4=st.slider("Sélectionner nombre de stations à afficher:",key=1,min_value=1,max_value=50)
    data_sample = data.sample(n=line_to_plot4, random_state=42)
    data_sample['Longitude'] = data_sample['Longitude'].str.replace(',', '.')
    data_sample['Latitude'] = data_sample['Latitude'].str.replace(',', '.')
    gdf_incident = gpd.GeoDataFrame(data_sample, geometry=gpd.points_from_xy(data_sample['Longitude'].astype(float), data_sample['Latitude'].astype(float)))
    
    m = folium.Map(location=[51.5074, -0.1278], zoom_start=10,tiles='CartoDB positron')
    marker = folium.Marker(
        location=[51.50335, -0.09862],
        popup='London Fire Brigade',
        icon=folium.Icon(icon="fire",prefix="fa",color='red'))
    marker.add_to(m)
    
    marker_cluster = MarkerCluster()

    for idx, row in gdf_incident.iterrows():
        attendance_time = row['AttendanceTimeSeconds']
        radius = attendance_time / 50
        tooltip = f"Nom : {row['ProperCase']}<br>" \
              f"Temps d'intervention : {row['AttendanceTimeSeconds']} secondes<br>" \
              f"Type de propriété : {row['PropertyCategory']}<br>" \
              f"Station : {row['IncidentStationGround']}<br>" \
              f"Arrondissement : {row['IncGeo_BoroughName']}<br>" \
              f"Quartier : {row['IncGeo_WardName']}"
        folium.CircleMarker(location=[row['Latitude'], row['Longitude']],radius=radius,color='red', tooltip=tooltip).add_to(marker_cluster)
    
    marker_cluster.add_to(m)
    st_data = st_folium(m, width=725)



elif page==pages[4]:  #Modélisation
    
    def train_model(model_choisi, X_train, y_train, X_test, y_test) :
        if model_choisi == 'Random Forest' : 
            model = RandomForestRegressor()
        elif model_choisi == 'Decision Tree' : 
            model = DecisionTreeClassifier()
        elif model_choisi == 'KNN' : 
            model = KNeighborsClassifier()
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        return score

    def demo_streamlit():
        
        st.markdown("<h1 style='text-align: center; font-size: 25px;'>Modélisation</h1>", unsafe_allow_html=True)

        st.markdown("<p style='font-size: 15px;'></p>", unsafe_allow_html=True)

        st.markdown(
        """
        Pour la modélisation nous avons effectué une sélection des données pertinentes, values = 
        - 'Hillingdon' 
        - 'Greenwich' 
        - 'Haringey'
        - 'Enfield'
        - 'Bromley'
        - 'Hounslow'
        - 'Waltham Forest'
        - 'Redbridge'
        - 'Havering'
        - 'Barking And dagenham'
        - 'Bexley'
        - 'Harrow'
        - 'Sutton'
        - 'Merton'
        - 'Richmond Upon thames'
        - 'Kingston Upon thames'
        - 'City Of london'
        """
        )
         
        st.markdown(
        """
        Ainsi que, values = 
        - 'Paddington' 
        - 'Soho' 
        - 'Euston'
        - 'Lambeth'
        - 'North Kensington'
        - 'Chelsea'
        - 'Shoreditch'
        - 'West Hampstead'
        - 'Croydon'
        - 'Stoke Newington'
        - 'Hammersmith'
        - 'Kentish Town'
        - 'Poplar'
        - 'Homerton'
        - 'Bethnal Green'
        - 'Brixton'
        """
        )
        
        st.markdown(
        """
        Suivi d'une répartition du temps en deux périodes:
        - 'Matin' et
        - 'Soir'
        """
        )
            
        st.markdown(
        """
        Ensuite nous avons réalisé une conversion de la colonne 'DateAndTimeMobilised' en temps et une encodage.
        """
        )
        
        df=pd.read_csv('df_new.csv')
        
        #Séléction des données pertinentes

        values = ['Hillingdon', 'Greenwich', 'Haringey', 'Enfield', 'Bromley', 'Hounslow', 'Waltham Forest',
                  'Redbridge', 'Havering', 'Barking And dagenham', 'Bexley', 'Harrow', 'Sutton', 'Merton',
                  'Richmond Upon thames', 'Kingston Upon thames', 'City Of london']
        
        df= df[df.ProperCase.isin(values) == False]
        
        df['ProperCase'].value_counts(normalize=True)
        
        values = ['Paddington', 'Soho', 'Euston', 'Lambeth', 'North Kensington', 'Chelsea', 'Shoreditch', 
                  'West Hampstead', 'Croydon', 'Stoke Newington', 'Hammersmith', 'Kentish Town', 
                  'Poplar', 'Homerton', 'Bethnal Green','Brixton']
        
        df = df[df.IncidentStationGround.isin(values) == True]
        
        df['IncidentStationGround'].value_counts(normalize=True)
        df['Périodes']=pd.cut(pd.to_datetime(df.DateAndTimeMobilised).dt.hour,
           bins=[0, 12, 23],
           labels=['Matin','Soir'],
           ordered=False,
           include_lowest=True)   
        
        # 4. Conversion de la colonne date en temps

        df['DateAndTimeMobilised'] = pd.to_datetime(df.DateAndTimeMobilised)
        df['Year'] = df['DateAndTimeMobilised'].dt.year
        df['Month'] = df['DateAndTimeMobilised'].dt.month
        df['Day'] = df['DateAndTimeMobilised'].dt.day
        df['Hour'] = df['DateAndTimeMobilised'].dt.hour
        
         # 5. Renommage des données et traduction en Français

        df = df.rename(columns = {'DateOfCall': 'Date_appel',
                         'TurnoutTimeSeconds':'Tps_preparation',
                         'IncidentStationGround':'Caserne',
                         'Year':'Année',
                         'PropertyCategory':'Type_propriété',
                         'ProperCase':'Nom_arrondissement',
                         'Month':'Mois',
                         'Hour':'Heure',
                         'AttendanceTimeSeconds': 'Tps_arrivée',
                         'DateAndTimeMobilised':'Date_mobilisation',
                          'PumpOrder':'Ordre_pompe'
                         })

        # 6. Encodage des données

        label_encoder = LabelEncoder()
        
        df["Périodes"] = label_encoder.fit_transform(df["Périodes"])
        df["Type_propriété"] = label_encoder.fit_transform(df["Type_propriété"])
        df["Caserne"] = label_encoder.fit_transform(df["Caserne"])
        df["Nom_arrondissement"] = label_encoder.fit_transform(df["Nom_arrondissement"])
        df["Date_mobilisation"] = label_encoder.fit_transform(df["Date_mobilisation"])
        
        y = df['Périodes']
        X = df.drop(['Date_mobilisation','Tps_arrivée','Date_appel','Tps_preparation',
                    'Périodes', 'Heure'], axis = 1)
        
        train_size = st.slider(label = "Choix de la taille de l'échantilllon de train", min_value = 0.2, max_value = 1.0, step = 0.05)
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = train_size, random_state =0)
            
       # 8. Normalisation des données 
    
        scaler = StandardScaler()
        
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        model = LogisticRegression() 
        model.fit(X_train, y_train)
        st.write("Précision régression logistique :" , model.score(X_test,y_test))
    
        
        model_list = ['Decision Tree', 'KNN','Random Forest']
        model_choisi = st.selectbox(label = "Sélectionner un modèle" , options = model_list)
    
    
        # Showing the accuracy for the orthers models (for comparison)
        st.write("Accuracy du modèle: ")
        st.write("Score test", train_model(model_choisi, X_train, y_train, X_test, y_test))    
        
        
    demo_streamlit()

elif page==pages[5]:
    
    st.markdown("<h1 style='text-align: center; font-size: 25px;'>Calcul des distances et application dummy</h1>", unsafe_allow_html=True)
    
    st.header("Calcul des distances")
    
    st.text("Base de données df_inc_dist :")
    
    line_to_plot3=st.slider("Sélectionner nombre de lignes à afficher:",key=1,min_value=3,max_value=100)
    st.dataframe(df_inc_dist.sample(line_to_plot3))
    
    st.text("Base de données df_mob_dist :")
    
    line_to_plot=st.slider("Sélectionner nombre de lignes à afficher:",key=2,min_value=3,max_value=100)
    st.dataframe(df_mob_dist.sample(line_to_plot))
    
    st.text("Base de données df_dist :")
    
    line_to_plot=st.slider("Sélectionner nombre de lignes à afficher:",key=3,min_value=3,max_value=100)
    st.dataframe(df_dist.sample(line_to_plot))
    
    st.text("Base de données df_coordo :")
    
    line_to_plot=st.slider("Sélectionner nombre de lignes à afficher:",key=4,min_value=3,max_value=100)
    st.dataframe(df_coordo.sample(line_to_plot))

    st.text("Base de données df_dista :")
    
    line_to_plot=st.slider("Sélectionner nombre de lignes à afficher:",key=5,min_value=3,max_value=100)
    st.dataframe(df_dista.sample(line_to_plot))
    
    st.header("Application dummy")
    
    def train_model(model_choisi, X_train, y_train, X_test, y_test) :
        if model_choisi == 'Random Forest' : 
            model = RandomForestRegressor()
        elif model_choisi == 'Decision Tree' : 
            model = DecisionTreeClassifier()
        elif model_choisi == 'KNN' : 
            model = KNeighborsClassifier()
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        return score

    def demo_streamlit():
        
        
        df=pd.read_csv('df_new.csv')
        
        #Séléction des données pertinentes

        values = ['Hillingdon', 'Greenwich', 'Haringey', 'Enfield', 'Bromley', 'Hounslow', 'Waltham Forest',
                  'Redbridge', 'Havering', 'Barking And dagenham', 'Bexley', 'Harrow', 'Sutton', 'Merton',
                  'Richmond Upon thames', 'Kingston Upon thames', 'City Of london']
        
        df= df[df.ProperCase.isin(values) == False]
        
        df['ProperCase'].value_counts(normalize=True)
        
        values = ['Paddington', 'Soho', 'Euston', 'Lambeth', 'North Kensington', 'Chelsea', 'Shoreditch', 
                  'West Hampstead', 'Croydon', 'Stoke Newington', 'Hammersmith', 'Kentish Town', 
                  'Poplar', 'Homerton', 'Bethnal Green','Brixton']
        
        df = df[df.IncidentStationGround.isin(values) == True]
        
        df['IncidentStationGround'].value_counts(normalize=True)
        df['Périodes']=pd.cut(pd.to_datetime(df.DateAndTimeMobilised).dt.hour,
           bins=[0, 12, 23],
           labels=['Matin','Soir'],
           ordered=False,
           include_lowest=True)   
        
        # 4. Conversion de la colonne date en temps

        df['DateAndTimeMobilised'] = pd.to_datetime(df.DateAndTimeMobilised)
        df['Year'] = df['DateAndTimeMobilised'].dt.year
        df['Month'] = df['DateAndTimeMobilised'].dt.month
        df['Day'] = df['DateAndTimeMobilised'].dt.day
        df['Hour'] = df['DateAndTimeMobilised'].dt.hour
        
         # 5. Renommage des données et traduction en Français

        df = df.rename(columns = {'DateOfCall': 'Date_appel',
                         'TurnoutTimeSeconds':'Tps_preparation',
                         'IncidentStationGround':'Caserne',
                         'Year':'Année',
                         'PropertyCategory':'Type_propriété',
                         'ProperCase':'Nom_arrondissement',
                         'Month':'Mois',
                         'Hour':'Heure',
                         'AttendanceTimeSeconds': 'Tps_arrivée',
                         'DateAndTimeMobilised':'Date_mobilisation',
                          'PumpOrder':'Ordre_pompe'
                         })

        # 6. Encodage des données

        label_encoder = LabelEncoder()
        
        df["Périodes"] = label_encoder.fit_transform(df["Périodes"])
        df["Type_propriété"] = label_encoder.fit_transform(df["Type_propriété"])
        df["Caserne"] = label_encoder.fit_transform(df["Caserne"])
        df["Nom_arrondissement"] = label_encoder.fit_transform(df["Nom_arrondissement"])
        df["Date_mobilisation"] = label_encoder.fit_transform(df["Date_mobilisation"])
        
        y = df['Périodes']
        X = df.drop(['Date_mobilisation','Tps_arrivée','Date_appel','Tps_preparation',
                    'Périodes', 'Heure','Mois','Année','Tps_preparation',"Caserne"], axis = 1)
        
        train_size = st.slider(label = "Choix de la taille de l'échantilllon de train", min_value = 0.2, max_value = 1.0, step = 0.05)
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = train_size, random_state =0)
            
       # 8. Normalisation des données 
    
        scaler = StandardScaler()
        
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        model = LogisticRegression() 
        model.fit(X_train, y_train)
        st.write("Précision régression logistique :" , model.score(X_test,y_test))
    
        
        model_list = ['Decision Tree', 'KNN','Random Forest']
        model_choisi = st.selectbox(label = "Sélectionner un modèle" , options = model_list)
    
    
        train_model(model_choisi, X_train, y_train, X_test, y_test)
        
        #Amb
        # clf_rf = ensemble.RandomForestClassifier()
        # rf_param = {
        #     'max_features': ['sqrt', 'log2'],
        #     'min_samples_split': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
        # }

        # grid_clf = model_selection.GridSearchCV(estimator=clf_rf, param_grid=rf_param)
        # grille = grid_clf.fit(X_train, y_train)

        # best_score = grille.best_score_
        # best_params = grille.best_params_

        # st.write("Best Score:", best_score)
        # st.write("Best Parameters:", best_params)
        
        
        if st.button("Prédire la Période"):
            model = None
            
            if model_choisi == 'Decision Tree':
                model = DecisionTreeClassifier()
            elif model_choisi == 'KNN':
                model = KNeighborsClassifier()
            elif model_choisi == 'Random Forest':
                model = RandomForestRegressor()
            
            model.fit(X_train, y_train)
            
         
        Typedepropriété = st.selectbox('Type de propriété', ("Outdoor", "Dwelling", "Non Residential", "Outdoor Structure", "Transport Vehicle", "Other Residential"))
        Nomarrondissement = st.selectbox('Nom arrondissement', ('Enfield', 'Barnet', 'Westminster', "Tower Hamlets", 'Newham', "Waltham Forest", 'Lambeth', 'Harrow', "Barking And dagenham", 'Hounslow', 'Islington', 'Bexley', 'Haringey', 'Camden', "Richmond Upon thames", 'Greenwich', 'Ealing', "Kingston Upon thames", "Hammersmith And fulham", 'Redbridge', 'Havering', 'Wandsworth', 'Hillingdon', 'Brent', 'Merton', 'Croydon', 'Southwark', 'Lewisham', 'Hackney', 'Bromley', "Kensington And chelsea", 'Sutton', "City Of london"))
        #Caserne=st.selectbox('Caserne',("Edmonton", "Barnet", "Kensington", "Poplar", "Stratford", "Walthamstow", "Clapham", "Stanmore", "North Kensington", "Dagenham", "Harrow", "Hendon", "Ealing", "Holloway", "Erith", "Hornsey", "Kentish Town", "Richmond", "Eltham", "Plumstead", "Southall", "New Malden", "Acton", "Barking", "Tottenham", "Plaistow", "Ilford", "Romford", "Southgate", "East Greenwich", "Feltham", "Battersea", "Whitechapel", "Heston", "Willesden", "Mitcham", "Chingford", "Croydon", "Old Kent Road", "Bromley", "Soho", "Hornchurch", "Fulham", "Paddington", "Peckham", "Shoreditch", "Surbiton", "Lambeth", "Bethnal Green", "Lewisham", "Stoke Newington", "Twickenham", "Wembley", "Hammersmith", "Wimbledon", "Mill Hill", "Ruislip", "Chiswick", "Enfield", "West Hampstead", "Sutton", "Sidcup", "Orpington", "Brixton", "Dowgate", "East Ham", "Bexley", "Norbury", "Beckenham", "Shadwell", "Addington", "Chelsea", "Kingston", "Leytonstone", "Hainault", "Heathrow", "Tooting", "New Cross", "Dockhead", "Forest Hill", "Hayes", "Park Royal", "Woodford", "West Norwood", "Lee Green", "Wallington", "Leyton", "Homerton", "Harold Hill", "Wandsworth", "Islington", "Northolt", "Euston", "Finchley", "Woodside", "Purley", "Hillingdon", "Deptford", "Millwall", "Wennington", "Biggin Hill", "Greenwich"))
        
        if st.button("Prédire"):
            input_data = [[Typedepropriété, Nomarrondissement]]
            input_df = pd.DataFrame(input_data, columns=['Type_propriété', 'Nom_arrondissement'])
            
            # Map selected values to their corresponding encoded numeric values
            type_propriete_mapping = {'Outdoor': 0, 'Dwelling': 1, 'Non Residential': 2, 'Outdoor Structure': 3, 'Transport Vehicle': 4, 'Other Residential': 5}
            nom_arrondissement_mapping = {'Enfield': 0, 'Barnet': 1, 'Westminster': 2, 'Tower Hamlets': 3, 'Newham': 4, 'Waltham Forest': 5, 'Lambeth': 6, 'Harrow': 7, 'Barking And dagenham': 8, 'Hounslow': 9, 'Islington': 10, 'Bexley': 11, 'Haringey': 12, 'Camden': 13, 'Richmond Upon thames': 14, 'Greenwich': 15, 'Ealing': 16, 'Kingston Upon thames': 17, 'Hammersmith And fulham': 18, 'Redbridge': 19, 'Havering': 20, 'Wandsworth': 21, 'Hillingdon': 22, 'Brent': 23, 'Merton': 24, 'Croydon': 25, 'Southwark': 26, 'Lewisham': 27, 'Hackney': 28, 'Bromley': 29, 'Kensington And chelsea': 30, 'Sutton': 31, 'City Of london': 32}

            input_df["Type_propriété"] = input_df["Type_propriété"].map(type_propriete_mapping)
            input_df["Nom_arrondissement"] = input_df["Nom_arrondissement"].map(nom_arrondissement_mapping)
        
            prediction = model.predict(input_df)
            st.write("Prédiction de la Période:", prediction)
   
         
        # Typedepropriété=st.selectbox('Type de propriété',("Outdoor","Dwelling","Non Residential","Outdoor Structure","Transport Vehicle","Other Residential"))
        # Nomarrondissement=st.selectbox('Nom arrondissement',('Enfield', 'Barnet', 'Westminster' ,'Tower Hamlets' ,'Newham'
        #  'Waltham Forest' ,'Lambeth' ,'Harrow' ,'Barking And dagenham' ,'Hounslow',
        #  'Islington' ,'Bexley' ,'Haringey' ,'Camden', 'Richmond Upon thames',
        #  'Greenwich', 'Ealing', 'Kingston Upon thames', 'Hammersmith And fulham',
        #  'Redbridge' ,'Havering' ,'Wandsworth' ,'Hillingdon', 'Brent' ,'Merton',
        #  'Croydon' ,'Southwark' ,'Lewisham', 'Hackney' ,'Bromley',
        #  'Kensington And chelsea' ,'Sutton' ,'City Of london'))
        # Result=""
            
        # if st.button("Prédire"):
        #     input_data = [[Typedepropriété,Nomarrondissement]]
        #     input_df = pd.DataFrame(input_data, columns=['Type_propriété', 'Nom_arrondissement'])
        #     input_df["Type_propriété"] = label_encoder.transform(input_df["Type_propriété"])
        #     input_df["Nom_arrondissement"] = label_encoder.transform(input_df["Nom_arrondissement"])
                
        #     prediction = model.predict(input_df)
        #     st.write("Prédiction de la Période:", prediction)
        
    demo_streamlit()

