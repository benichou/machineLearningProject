"""===========================================================================
Travail Machine Learning

==========================================================================="""
# CHOSES A FAIRE:
# - Sortir 30 articles aléatoirement (FRANCK)
# - Trouver comment conserver les accents dans le pre-processing (FRANCK)
# - Visualisation avec LDA (FRANCK)
# - Créer les second niveaux de cluster (3 niveaux au total) (GAB, SIMON)
# - Faire un petit dataframe de statistique des cluster (GAB)

# NEXT STEP 
# - 


#------------------------------------------------------------------------------
# Importation des libraries
import pandas as pd
import numpy as np
import nltk
import re
import unidecode
import random
import statistics as stat
import os
from time import time
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer, PorterStemmer, LancasterStemmer
from nltk.tokenize import RegexpTokenizer
from time import time
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction import text 
from sklearn.cluster import KMeans
from langdetect import detect
from sklearn.decomposition import NMF, LatentDirichletAllocation

# Au besoin
#nltk.download()
#------------------------------------------------------------------------------
"""===================================================
Etape 1 : IMPORTATION DES DONNÉES
==================================================="""

### Définir les répertoires de données:
# Gabriel
raw_metadata = pd.read_csv("C:/projects/machineLearning/hecCourse/metadata_lang.csv")
# Simon
# raw_metadata = pd.read_csv("D:\Dropbox\Apprentiassage Automatique I\Code\metadata\metadata_lang.csv")

# =============================================================================
# # Retirer les doublons
# metadata = raw_metadata.drop_duplicates(subset=['cord_uid'])
# # duplic = raw_metadata[raw_metadata['cord_uid'] == 'suvag5m2'] # TEST
# 
# # Identifier articles sans abstract et les retirer
# NAs = pd.isna(raw_metadata['abstract']) 
# metadata = raw_metadata[np.invert(NAs)]
# 
# # Retirer les articles ayant un abstract de moins de 100 caractères
# metadata = metadata[metadata['abstract'].map(len) > 100]
# 
# # Reset les index metadata
# metadata = metadata.reset_index(drop=True)
# =============================================================================


# Extract ID and abstract from dataset
# !!! Ajouter filtre pour articles en anglais

df_all_abstracts = raw_metadata.loc[raw_metadata['langage']=='en'][['cord_uid', 'abstract']]
df_abstracts = raw_metadata.loc[raw_metadata['langage']=='en'][['cord_uid', 'abstract']][200000:200500]  #Nombre d'abstracts sélectionnés


# SECTION PARAMÈTRES 
# Paramètres niveau 1
n_clust_level1 = 3
n_bestwords_level1 = 25
# Paramètres niveau 2
n_clust_level2 = 3
n_bestwords_level2 = 10
# Paramètres niveau 3
n_clust_level3 = 3
n_bestwords_level3 = 10



"""===================================================
FUNCTIONS
==================================================="""

# =============================================================================
# # Identifier langage des abstracts
# start = time()
# language1 = []
# for i in metadata['abstract'][0:50000]: # Par bloque de 50 000 obs
#     language1.append(detect(i))
# stop = time()
# stop-start
# =============================================================================


#-------------------- FUNCTION remove_accents_numbers() -----------------------
def remove_accents_numbers(text):
    """Retirer les accents et les chiffres d'un texte, e.g. café -> cafe, 
    'En 1984 dans les journaux' -> 'En   dans les journaux'
    Argument : string
    """
    pattern1 = '[0-9-]'
    text = unidecode.unidecode(text)
    text = re.sub(pattern1,'',text)
    return text

#----------------- FUNCTION bestwords_by_cluster_kmeans() ---------------------
def bestwords_by_cluster_kmeans(n_clust, n_bestwords, tfidf_matrix, scv ):
    """
    Parameters
    ----------
    n_clust : TYPE: Integer
        DESCRIPTION: Nombre de cluster à produire
    n_bestwords : TYPE: Integer
        DESCRIPTION: Nombre de mots à retenir par cluster
    tfidf_matrix : TYPE: sparse matrix
        DESCRIPTION. Sparse matrix contenant la matrice TF-IDF
    scv : TYPE: object StemmedCountVectorizer
        DESCRIPTION: Initialisation de la fonction StemmedCountVectorizer

    Returns
    -------
    bestwords_by_cluster : TYPE: DataFrame.
        DESCRIPTION: Tableau des n_bestwords par cluster.
    kmeans_model : TYPE: Model K-mean.
    listlist_bw: TYPE: List.
        DESCRIPTION: List des lists de bestwords par cluster.
    df_word_rep: TYPE: DataFrame.
        DESCRIPTION: Nombre de bestwords non unique de chaque cluster.
    df_stats: TYPE: DataFrame.
        DESCRIPTION: Quelques statistiques du modèle. 
    """
    # Initiatialiser le K-means model
    kmeans_model = KMeans(n_clusters=n_clust, random_state=0)
    # Fitter le K-means model
    kmeans_model.fit(tfidf_matrix)
    # Ordonner l'importance des mots par cluster 
    ordered_terms_kmeans = kmeans_model.cluster_centers_.argsort()[:, ::-1]
    terms_tfidf = scv.get_feature_names() # Extraire le label des features
    # Création du array des n_bestwords
    listlist_bw = []
    for i in range(n_clust):
        temp_list = []
        for position in ordered_terms_kmeans[i,0:n_bestwords]:
            temp_list.append(terms_tfidf[position])
        listlist_bw.append(temp_list)
    col_name = []
    for i in range(len(listlist_bw)):
        col_name.append("Cluster "+str(i+1))
    arr = np.array(listlist_bw)
    bestwords_by_cluster = pd.DataFrame(arr.T, columns = col_name)

    word_rep_list = []
    for list_i in listlist_bw: # pour toutes les listes i
        word_rep = 0 # initialiser le score pour la liste i
        for list_j in listlist_bw: # pour toutes les listes j
            if list_i != list_j: # si les liste sont différentes
                for word in list_i: # pour chaque mot de la liste i
                    if word in list_j: # si ce mots ets dans la liste j
                        word_rep += 1 # ajouter 1 au score de la liste i
        word_rep_list.append(word_rep) # appender le score de la liste i

    unrolded_list = [word for sublist in listlist_bw for word in sublist]
    nbr_bw_allclusters = len(unrolded_list)
    unique_bestwords = list(set(unrolded_list))
    nbr_uniq_bestwords = len(unique_bestwords)
    
    # Nombre de répétition en pourcentage
    perct_rep_list = np.round((np.array(word_rep_list) / (nbr_uniq_bestwords))*100, 2)
    # Nombre de répétitions relatives moyennes dans l'ensemble des clusters
    average_perct_rep = round(stat.mean(perct_rep_list),2)
    # Proportion de mots uniques dans tous les clusters
    prop_uniq_bw = round(nbr_uniq_bestwords/nbr_bw_allclusters*100, 2)
    
    # Mettre le tout dans un dataframe
    df_word_rep = pd.DataFrame([word_rep_list, perct_rep_list], 
                            columns = col_name,
                            index = ['Number rep', '% of words'])
    df_stats = pd.DataFrame([n_clust, n_bestwords, nbr_bw_allclusters, nbr_uniq_bestwords, average_perct_rep, prop_uniq_bw ], 
                            columns = ['Stats of this clustering'],
                            index = ['nbr clusters','nbr bestwords by cluster','nbr bestwords overall', 'nbr uniq bestwords', 'average percent word rep (%)', 'prop uniq bestwords (%)'])
    # Pour afficher le dataframe en entier
    #pd.set_option('display.max_rows', None)
    #pd.set_option('display.max_columns', None)
    #pd.set_option('display.width', None)
    #print(df_word_rep)
    #print(df_stats)
    #return df_word_rep, df_stats
    return bestwords_by_cluster, kmeans_model, df_word_rep, df_stats

#----------------- FUNCTION bestwords_by_cluster_lda() ---------------------
def bestwords_by_cluster_lda(n_clust, n_bestwords, tfidf_matrix, scv ):
    """
    Parameters
    ----------
    n_clust : TYPE: Integer
        DESCRIPTION: Nombre de cluster à produire
    n_bestwords : TYPE: Integer
        DESCRIPTION: Nombre de mots à retenir par cluster
    tfidf_matrix : TYPE: sparse matrix
        DESCRIPTION. Sparse matrix contenant la matrice TF-IDF
    scv : TYPE: object StemmedCountVectorizer
        DESCRIPTION: Initialisation de la fonction StemmedCountVectorizer

    Returns
    -------
    bestwords_by_cluster : TYPE: DataFrame
        DESCRIPTION: Tableau des n_bestwords par cluster
    kmeans_model : TYPE
        DESCRIPTION.

    """
    # Initiatialiser le K-means model
    lda_model = LatentDirichletAllocation(n_components=n_clust, random_state=0, max_iter=10)
    # Fitter le LDA model
    lda_fit_transform = lda_model.fit_transform(tfidf_matrix)
    #Déterminer les classes les plus probables pour chaque abstract
    lda_classes = lda_fit_transform.argmax(axis=1)
    # "Sort" de l'importance des mots par cluster 
    ordered_terms_lda = lda_model.components_.argsort()[:, ::-1]
    terms_count = scv.get_feature_names()


    # Création du array des n_bestwords
    all_list = []
    for i in range(n_clust):
        temp_list = []
        for position in ordered_terms_lda[i,0:n_bestwords]:
            temp_list.append(terms_count[position])
        all_list.append(temp_list)
    col_name = []
    
    for i in range(len(all_list)):
        col_name.append("Cluster_"+str(i+1))
    arr = np.array(all_list)

    bestwords_by_cluster = pd.DataFrame(arr.T, columns = col_name)
    return bestwords_by_cluster, lda_model, lda_classes


#--------------------  Classe : StemmedCountVectorizer ------------------------
# Classe : StemmedCountVectorizer
stemmer = SnowballStemmer('english') # Définir le stemmer :
# Ajouter un stemmer à la fonction "CountVectorizer" :
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])



"""===================================================
Etape 1 : PREPROCESS
==================================================="""
# Converti les accents en non accents
df_abstracts['abs_clean'] = df_abstracts.apply(lambda x: remove_accents_numbers(x['abstract']), axis=1) 


# AJOUTER STOPWORDS
#print(text.ENGLISH_STOP_WORDS) # Voir la liste des STOPWORDS
new_stopwords = [] # Ajouter des STOPWORDS à la liste de defaut
my_stopwords_list = text.ENGLISH_STOP_WORDS.union(new_stopwords) # Définir la nouvelle liste

"""=========================================================
Etape 2 : RÉCUPÉRATION DES ARTICLES A LIRE (une seule fois)
========================================================="""

# =============================================================================
# # Fonction pour sélectionner 20 cord_uid alléatoirement
# list_abstract_id = random.sample(list(df_all_abstracts['cord_uid']), 20)
# len(list_abstract_id)
# 
# # Récupérer les abstracts des 20 cord_uid:
# list_abst = []
# for i in list_abstract_id:
#     for j in list(df_all_abstracts['cord_uid']):        
#         if i == j:
#             temp = list(df_all_abstracts[df_all_abstracts['cord_uid'] == i]['abstract'])
#             list_abst.append(temp)
# 
# # Assembler dataframe                      
# df_toread = pd.DataFrame(list_abst, columns = ['abstract'])
# df_toread['cord_uid'] = list_abstract_id
# df_toread = df_toread[['cord_uid', 'abstract']]
# 
# # Save dataFrame to Excel
# save_path = 'D:/Dropbox/_HEC/Machine Learning 1/_Travaux Machine Learning 1/Code/ArticlesToLabel/abstract_to_read.xlsx'
# df_toread.to_excel(save_path, index=False)
# =============================================================================

#------------------------------------ TEST ------------------------------------
# Lire fichier Excel des label

# Gabriel
path = 'D:/Dropbox/_HEC/Machine Learning 1/_Travaux Machine Learning 1/Code/ArticlesToLabel/abstract_to_read.xlsx'
# Simon
#path = "D:\Dropbox\Apprentiassage Automatique I\Code/ArticlesToLabel/abstract_to_read_v2.xlsx"
df_to_read = pd.read_excel(path)

# Faire un string de mots pour chaque abstract
df_to_read['labels'] = df_to_read['label1']+' '+df_to_read['label2']+' '+df_to_read['label3']+' '+df_to_read['label4']+' '+df_to_read['label5']

# Retirer les nombre et les accents
df_to_read['labels_clean'] = df_to_read.apply(lambda x: remove_accents_numbers(x['labels']), axis=1) 

# Appliquer le stemming et la tokenisation
scv_label = StemmedCountVectorizer(strip_accents = 'ascii',lowercase = True ,stop_words= my_stopwords_list  ,analyzer='word')
list_token_label = []
for i in df_to_read['labels_clean']:
    scv_label.fit_transform([i])
    list_token_label.append(scv_label.get_feature_names())

df_to_read['labels_stemmed'] = list_token_label
df_to_read = df_to_read.drop(['label1', 'label2', 'label3', 'label4', 'label5'], axis=1 )

#-----------------------------------------------------------------------------

"""=========================================================
Etape 3 : Création matrice de fréquence et matrice TF-IDF
========================================================="""

# 1- Initialiser la fonction StemmedCountVectorizer 
scv = StemmedCountVectorizer(strip_accents = 'ascii'# Retirer les accents sur les lettres
                             ,lowercase = True      # Retirer les majuscules 
                             #,stop_words='english'  # Retirer les STOPWORDS à partir d'une liste
                             ,stop_words= my_stopwords_list  # Retirer les STOPWORDS à aprtir d'une liste
                             ,analyzer='word'
                             #,min_df=2  # Élimine les mots qui apparaissent dans moins de 2 documents
                             #,max_df=0.70  # Élimine les mots qui apparaissent dans plus de 75% des documents
                             )

# 2- Créer la matrice "count_matrix" avec le décompte de chaque mot par document
count_matrix = scv.fit_transform(df_abstracts['abs_clean'])
feature_names = scv.get_feature_names() # Extraire les noms de features

# =============================================================================
# # (PEUT ÊTRE COMMENTÉ) 
# # Voir la liste des mots de la matrice
# scv.vocabulary_
# # Voir la liste des STOPSWORDS du modèle
# scv.stop_words_
# # Voir la shape de la matrice : N documents X M words
# count_matrix.shape
# =============================================================================

# 3- Initialiser le transformer TF-IDF (Transform a count matrix to a normalized tf or tf-idf representation)
tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True) 
# 4- Calcul des valeurs IDF:
tfidf_transformer.fit(count_matrix)
# 4- Créer la matrice de score TF-IDF
tfidf_matrix = tfidf_transformer.transform(count_matrix)

# =============================================================================
# # (PEUT ÊTRE COMMENTÉ) Imprimer le score TF-IDF du premier document 
# feature_names = scv.get_feature_names() # Extraire les noms de features
# first_document_vector=tfidf_matrix[0] # Obtenir le vecteur TF-IDF du premier document  
# df = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"]) 
# df.sort_values(by = ["tfidf"],ascending=False)
# =============================================================================

"""=========================================================
Etape 4 : Clustering avec K-Means
========================================================="""
# Voir: https://stackoverflow.com/questions/50827222/understanding-and-applying-k-means-clustering-for-topic-modeling

#==============================================
# CLUSTERS NIVEAU 1
#==============================================
# lancer la fonction bestwords_by_cluster_kmeans() sur la matrice TF-IDF du nivau 1
kmeans_level1 = bestwords_by_cluster_kmeans(n_clust_level1, n_bestwords_level1, tfidf_matrix, scv)

# Ajouter les clusters correspondant à chaque article dans le dataframe original pour le niveau 1 
df_abstracts["kmeans_level1"] = list(kmeans_level1[1].labels_)

# Dictionnaire bestwords kmeans level 1
dict_best_words_kmeans_level1 = kmeans_level1[0] 


def isolate(level, cluster): #!!! RETIRER DU FORMAT FONCTION (SIMON)
    """
    level : string du nom de la colonne indiquant à quel niveau appartiennent 
    les clusters.
    cluster : entier représentant le numéro du cluster. 
    """
    return df_abstracts.loc[df_abstracts[level]==cluster, ['cord_uid', 'abs_clean']]


# Dictionnaire contenant les identifiants de cluster par article pour le level 1
dict_uniq_cluster_kemans_level1= {} # !!! Indiquer le nom du level dans le nom (dict_uniq_cluster_kemans_level1_level1)
for i in range(n_clust_level1):
    dict_uniq_cluster_kemans_level1['level1_cluster_'+str(i+1)] = isolate('kmeans_level1', i)

#==============================================
# CLUSTERS NIVEAU 2
#==============================================
# Création des clusters level2

# 2- Créer la matrice "count_matrix" avec le décompte de chaque mot par document
dict_best_words_kmeans_level2 = {}
list_count_matrix = []
list_tfidf_matrix = []
list_level2_clusters = []


#On crée les dictionnaires de clusters et de meilleurs mots
for index, (key,value) in enumerate(dict_uniq_cluster_kemans_level1.items()):
    one_count_matrix = scv.fit_transform(dict_uniq_cluster_kemans_level1[key]['abs_clean'])
    list_count_matrix.append(one_count_matrix)
    tfidf_transformer.fit(one_count_matrix)
    tfidf_matrix = tfidf_transformer.transform(one_count_matrix)
    list_tfidf_matrix.append(tfidf_matrix)
    kmeans_level2 = bestwords_by_cluster_kmeans(n_clust_level2, n_bestwords_level2, tfidf_matrix, scv)
    dict_uniq_cluster_kemans_level1[key]['cluster_level2'] = kmeans_level2[1].labels_
    dict_best_words_kmeans_level2['level2_cluster_'+str(index)] = kmeans_level2[0]


#On crée la liste contenant l'ensemble des labels triés par index
for i in dict_uniq_cluster_kemans_level1:
    if len(list_level2_clusters) == 0:
        list_level2_clusters = dict_uniq_cluster_kemans_level1[i]['cluster_level2']
    else:
        list_level2_clusters = list_level2_clusters.append(dict_uniq_cluster_kemans_level1[i]['cluster_level2'])
list_level2_clusters = list_level2_clusters.sort_index()
        
# Ajouter les clusters correspondant à chaque article dans le dataframe original pour le niveau 2
df_abstracts['kmeans_level2'] = list_level2_clusters


# Dictionnaire contenant les identifiants de cluster par article pour le level 2                  
dict_uniq_cluster_kmeans_level2= {}
for i in range(n_clust_level1):
    for j in range(n_clust_level2):
        dict_uniq_cluster_kmeans_level2['cluster_'+str(i+1)+'_'+str(j+1)] = df_abstracts.loc[((df_abstracts['kmeans_level1']==i) & (df_abstracts['kmeans_level2']==j)), ['cord_uid', 'abs_clean']]

#==============================================
# CLUSTERS NIVEAU 3
#==============================================
# Création des clusters level3

# Créer la matrice "count_matrix" avec le décompte de chaque mot par document
dict_best_words_kmeans_level3 = {}
list_count_matrix = []
list_tfidf_matrix = []
list_level3_clusters = []

#On crée les dictionnaires de clusters et de meilleurs mots
for index, (key,value) in enumerate(dict_uniq_cluster_kmeans_level2.items()):
    one_count_matrix = scv.fit_transform(dict_uniq_cluster_kmeans_level2[key]['abs_clean'])
    list_count_matrix.append(one_count_matrix)
    tfidf_transformer.fit(one_count_matrix)
    tfidf_matrix = tfidf_transformer.transform(one_count_matrix)
    list_tfidf_matrix.append(tfidf_matrix)
    kmeans_level3 = bestwords_by_cluster_kmeans(n_clust_level3, n_bestwords_level3, tfidf_matrix, scv)
    dict_uniq_cluster_kmeans_level2[key]['cluster_level3'] = kmeans_level3[1].labels_
    dict_best_words_kmeans_level3['level3_cluster_'+str(index)] = kmeans_level3[0]
    
    
#On crée la liste avec l'ensemble des labels triés par index
for i in dict_uniq_cluster_kmeans_level2:
    if len(list_level3_clusters) == 0:
        list_level3_clusters = dict_uniq_cluster_kmeans_level2[i]['cluster_level3']
    else:
        list_level3_clusters = list_level3_clusters.append(dict_uniq_cluster_kmeans_level2[i]['cluster_level3'])
list_level3_clusters = list_level3_clusters.sort_index()    

# Ajouter les clusters correspondant à chaque article dans le dataframe original pour le niveau 2
df_abstracts['kmeans_level3'] = list_level3_clusters 

             
"""=========================================================
Etape 4 : Clustering avec LDA (Latent Dirichlet Allocation)
========================================================="""    
# Voir: https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py

start = time()
# Rééinitialiser la matrice "count_matrix" avec le décompte de chaque mot par document
count_matrix = scv.fit_transform(df_abstracts['abs_clean'])

# Réétinitialiser la matrice TF-IDF
tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True) 
# Calcul des valeurs IDF:
tfidf_transformer.fit(count_matrix)
# Créer la matrice de score TF-IDF
tfidf_matrix = tfidf_transformer.transform(count_matrix)

lda_model = LatentDirichletAllocation(n_components=3, random_state=0, max_iter=5)
lda_model.fit_transform(tfidf_matrix)


# =============================================================================
# # Pour faire ressortir les termes les plus frequents dans les k clusters
# ordered_terms_lda = lda_model.components_.argsort()[:, ::-1]
# terms_count = scv.get_feature_names()
# 
# 
# print("Resultats clustering LDA \n")
# for i in range(3):
#     print("Cluster", i+1)
#     for position in ordered_terms_lda[i,0:10]:
#         print(terms_count[position]),
#     print ("\n")
# 
# stop = time()
# print(stop-start)
# =============================================================================

# Lancer la fonction bestwords_by_cluster_lda() sur la matrice TF-IDF du nivau 1
lda_level1 = bestwords_by_cluster_lda(n_clust_level1, n_bestwords_level1, tfidf_matrix, scv)

# Dictionnaire bestwords LDA level 1
dict_best_words_lda_level1 = lda_level1[0] 

# Ajouter les clusters correspondant à chaque article dans le dataframe original pour le niveau 1
df_abstracts["lda_level1"] = list(lda_level1[2])

# Dictionnaire contenant les identifiants de cluster par article pour le level 1
dict_uniq_cluster_lda_level1= {}
for i in list(range(n_clust_level1)):
    dict_uniq_cluster_lda_level1['cluster_'+str(i+1)] = df_abstracts.loc[df_abstracts['lda_level1']==i, ['cord_uid', 'abs_clean']]


#==============================================
# CLUSTERS NIVEAU 2
#==============================================
# Création des clusters level2

# Créer la matrice "count_matrix" avec le décompte de chaque mot par document
dict_best_words_lda_level2 = {}
list_count_matrix = []
list_tfidf_matrix = []
list_lda_level2_clusters = []


#On crée les dictionnaires de clusters et de meilleurs mots
for index, (key,value) in enumerate(dict_uniq_cluster_lda_level1.items()):
    one_count_matrix = scv.fit_transform(dict_uniq_cluster_lda_level1[key]['abs_clean'])
    list_count_matrix.append(one_count_matrix)
    tfidf_transformer.fit(one_count_matrix)
    tfidf_matrix = tfidf_transformer.transform(one_count_matrix)
    list_tfidf_matrix.append(tfidf_matrix)
    lda_level2 = bestwords_by_cluster_lda(n_clust_level2, n_bestwords_level2, tfidf_matrix, scv)
    dict_uniq_cluster_lda_level1[key]['cluster_level2'] = lda_level2[2]
    dict_best_words_lda_level2['level2_cluster_'+str(index)] = lda_level2[0]

#On crée la liste avec l'ensemble des labels triés par index
for i in dict_uniq_cluster_lda_level1:
    if len(list_lda_level2_clusters) == 0:
        list_lda_level2_clusters = dict_uniq_cluster_lda_level1[i]['cluster_level2']
    else:
        list_lda_level2_clusters = list_lda_level2_clusters.append(dict_uniq_cluster_lda_level1[i]['cluster_level2'])
list_lda_level2_clusters = list_lda_level2_clusters.sort_index()
        

# Ajouter les clusters correspondant à chaque article dans le dataframe original pour le niveau 2
df_abstracts['lda_level2'] = list_lda_level2_clusters

# Dictionnaire contenant les identifiants de cluster par article pour le level 2          
dict_uniq_cluster_lda_level2= {}
for i in range(n_clust_level1):
    for j in range(n_clust_level2):
        dict_uniq_cluster_lda_level2['cluster_'+str(i+1)+'_'+str(j+1)] = df_abstracts.loc[((df_abstracts['lda_level1']==i) & (df_abstracts['lda_level2']==j)), ['cord_uid', 'abs_clean']]


#==============================================
# CLUSTERS NIVEAU 3
#==============================================
# Création des clusters level3

# Créer la matrice "count_matrix" avec le décompte de chaque mot par document
dict_best_words_lda_level3 = {}
list_count_matrix = []
list_tfidf_matrix = []
list_lda_level3_clusters = []

#On crée les dictionnaires de clusters et de meilleurs mots
for index, (key,value) in enumerate(dict_uniq_cluster_lda_level2.items()):
    one_count_matrix = scv.fit_transform(dict_uniq_cluster_lda_level2[key]['abs_clean'])
    list_count_matrix.append(one_count_matrix)
    tfidf_transformer.fit(one_count_matrix)
    tfidf_matrix = tfidf_transformer.transform(one_count_matrix)
    list_tfidf_matrix.append(tfidf_matrix)
    lda_level3 = bestwords_by_cluster_lda(n_clust_level3, n_bestwords_level3, tfidf_matrix, scv)
    dict_uniq_cluster_lda_level2[key]['cluster_level3'] = lda_level3[2]
    dict_best_words_lda_level3['level3_cluster_'+str(index)] = lda_level3[0]
    
    
#On crée la liste avec l'ensemble des labels triés par index
for i in dict_uniq_cluster_lda_level2:
    if len(list_lda_level3_clusters) == 0:
        list_lda_level3_clusters = dict_uniq_cluster_lda_level2[i]['cluster_level3']
    else:
        list_lda_level3_clusters = list_lda_level3_clusters.append(dict_uniq_cluster_lda_level2[i]['cluster_level3'])
list_lda_level3_clusters = list_lda_level3_clusters.sort_index()    

# Ajouter les clusters correspondant à chaque article dans le dataframe original pour le niveau 2
df_abstracts['lda_level3'] = list_lda_level3_clusters


"""=========================================================
Etape 6 : Mesures de performance du clustering
========================================================="""

print(df_to_read)






"""==========================================================
Etape Non-negative Matrix Factorization (NMF)
=========================================================="""

# 1. Family of linear algebra algorithms for identifying the latent structure in data represented as a non-negative matrix.
# 2. NMF can be applied for topic modeling, where the input is term-document matrix, typically TF-IDF normalized.
# 3. Input: Term-Document matrix, number of topics.
# 4. Output: Two non-negative matrices of the original n words by k topics and those same k topics by the m original documents.
# 5 Basically, we are going to use linear algebra for topic modeling.

numTopics = 8

# alpha=0 means no regularization, l1_ratio=.5, the penalty is a combination of L1 and L2
nmf = NMF(n_components=numTopics, random_state=1, alpha=.1, l1_ratio=.5).fit(tfidf_matrix)
nmf_output = nmf.fit_transform(tfidf_matrix)





