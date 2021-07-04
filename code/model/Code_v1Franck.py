"""===========================================================================
Travail Machine Learning

==========================================================================="""
# CHOSES A FAIRE:
# - Sortir 30 articles aléatoirement (FRANCK)
# - Trouver comment conserver les accent dans le pre-processing (FRANCK)
# - Visualisation avec LDA
# - Créer les second niveaux de cluster (3 niveaux au total) (GAB SIMON)
# - Faire un petit dataframe de statistique des cluster ()


#------------------------------------------------------------------------------
# Importation des libraries
import pandas as pd
import numpy as np
import nltk
import re
import random
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer, PorterStemmer, LancasterStemmer
from nltk.tokenize import RegexpTokenizer
from time import time
import pandas as pd 
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.feature_extraction.text import CountVectorizer

import unidecode

# Au besoin
#nltk.download()
#------------------------------------------------------------------------------

### Définir le répertoire des données:
# Gabriel
raw_metadata = pd.read_csv("C:/projects/machineLearning/hecCourse/metadata.csv")
# Simon
#raw_metadata = pd.read_csv("D:\Dropbox\Apprentiassage Automatique I\Code\metadata.csv\metadata.csv")
# Franck
raw_metadata = pd.read_csv("C:/projects/machineLearning/hecCourse/metadata.csv")



#Basic Sandbox
import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

#To generate a refid for each paper (dataset + bib_entries)
import hashlib #for sha1

# #To build network and compute pagerank
import networkx as nx
import math as math

#For Data viz
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import date
from datetime import timedelta

#1. Get the data
datafiles = []
for dirname, _, filenames in os.walk('C:/projects/machineLearning/document_parses'):
    for filename in filenames:
            ifile = os.path.join(dirname, filename)
            if ifile.split(".")[-1] == "json":
                datafiles.append(ifile)

print("Number of Files Loaded: ", len(datafiles))

authors = []

citationsFlat = []
citationsCount = 0

for file in tqdm(datafiles):
    with open(file,'r')as f:
        doc = json.load(f)
    paper_id = doc['paper_id']
    
    paper_authors = []

    for value in doc['metadata']['authors']:
        if len(doc['metadata']['authors']) == 0:
            paper_authors.append("NA")
        else:
            last = value["last"]
            first = value["first"]
            paper_authors.append(first+" "+last)

    authors.append({"paper_id": paper_id, "authors" : paper_authors})

    for key,value in doc['bib_entries'].items():
        refid = key
        title = value['title'].lower()
        year = value['year']
        venue = value['venue'] 
        SHATitleCitation = hashlib.sha1(title.lower().encode()).hexdigest() #

        if (len(title) == 0):
            continue #there is noting we can do without any title

        citationsFlat.append({"citationId":citationsCount,\
                          "refid" : SHATitleCitation,\
                          "from": paper_id,\
                          "title": title.lower(),\
                          "year": year})
        citationsCount=citationsCount+1      


#Conversion into DataFrame
dfCitationsFlat = pd.DataFrame(citationsFlat)
authorsDf = pd.DataFrame(authors)
metadata = raw_metadata

metadata_extract = metadata[["cord_uid","sha", "title", "abstract", "publish_time"]].rename(columns = {"sha" : "paper_id"})
dfPaperList = pd.merge(metadata_extract, authorsDf, on = "paper_id", how = "left")

dfPaperList["year"] = 0
dfPaperList["refid"] = ""

for i in tqdm(range(len(dfPaperList))):
    
    dfPaperList["refid"][i] =  hashlib.sha1(str(dfPaperList["title"][i]).lower().encode()).hexdigest()
     #NB: We are building a custom identifier based on papers titles to ensure identification will be consistent between the papers in the Research Dataset and the papers extracted from the bib entries.
     #Unfortunately a paperId is not present for citations and doi is not provided for the whole dataset but title seem to be present for ~98% of the dataset. To enable and ease indexing capabilities we are hashing with SHA   
    dfPaperList["year"][i] = str(dfPaperList["publish_time"][i])[:4]
    
    try:
        dfPaperList["authors"][i] = dfPaperList["authors"][i].split(";")
    except:
        continue
        
quotationPapersFreq = pd.DataFrame({"refid" : dfCitationsFlat["refid"].value_counts().index, 
                       "nbQuotations" : dfCitationsFlat["title"].value_counts().values}) 

paperToScore = pd.merge(dfPaperList,quotationPapersFreq, on = "refid", how = "left")
paperToScore["nbQuotations"] = paperToScore["nbQuotations"].fillna(0)


#Adding list of references by papers according to the refid
refList = pd.DataFrame({"references" : dfCitationsFlat.groupby('from')['refid'].apply(list)}) 
refList["paper_id"] = refList.index; cols = ["paper_id","references"] ; refList = refList[cols].reset_index(drop = True) #Reformatting the reflist by papers
datasetForScoring = pd.merge(paperToScore, refList, how='left', on = 'paper_id').reset_index(drop = True)

datasetForScoring = datasetForScoring[(datasetForScoring["authors"].isna() == False)].reset_index(drop = True)

# datasetForScoring = pd.merge(datasetForScoring, metadata_extract[['cord_uid', 'paper_id']], on = 'paper_id', how='inner')

datasetForScoring['nbQuotations'] = pd.to_numeric(datasetForScoring['nbQuotations'], errors='coerce') # ensure that all the values in the nbQuotation columns are numerical values. If Null, they are then set to 0


#1. Creating an author dataset + Computation of the author page rank using an author network
#Variables for author dataset: id, name, co-authors, number of points linked to quotations, paper_count, citations, average citations,co_author_avg_citations,h-index

# author_data = {}
# author_id = {
#     'start': 1,
#     'curr': 1
# }

# assigned_ids = {}

# def create_author_data(train_data, author_data, author_id, assigned_ids):
#     for i in tqdm(range(len(train_data))):
#         authors = train_data.authors[i]
    
#         try:
#             citations = train_data.nbQuotations[i]/len(authors) #Number of times a paper have been quoted divided by len authors
#         except:
#             continue

#         for author in authors:
#             names = author.split(' ')
#             unique_name = names[0] + "_" + names[len(names)-1]
#             if unique_name not in author_data:
#                 author_data[unique_name] = {
#                     'num_citations': citations,
#                     'paper_count': 1,
#                     'name': unique_name,
#                     'author_id': author_id['curr'],
#                     'co_authors': {},
#                     'citations': [train_data.nbQuotations[i]]
#                 }
#                 assigned_ids[unique_name] = author_id['curr']
#                 author_id['curr'] += 1

#             else:
#                 author_data[unique_name]['num_citations'] += citations
#                 author_data[unique_name]['paper_count'] += 1
#                 author_data[unique_name]['citations'].append(train_data.nbQuotations[i])

#             for co_author in authors:
#                 co_author_names = co_author.split(' ')
#                 co_author_unique_name = co_author_names[0] + "_" + co_author_names[len(co_author_names)-1]
#                 if co_author_unique_name != unique_name:
#                     author_data[unique_name]['co_authors'][co_author_unique_name] = 1
                        
            
            
# # call for each data file
# create_author_data(datasetForScoring, author_data, author_id, assigned_ids)

# # add average citations
# for data in tqdm(author_data):
#     author_data[data]['average_citations'] = author_data[data]['num_citations'] / author_data[data]['paper_count']
    
# # adding h-index
# def get_h_index(citations):
#     return ([0] + [i + 1 for i, c in enumerate(sorted(citations, reverse = True)) if c >= i + 1])[-1]

# data_to_df = []
# for data in tqdm(author_data):
#     each_author = author_data[data]
#     co_authors = each_author['co_authors']
#     co_author_ids = []
#     co_author_avg_citations = 0
#     for co_author in co_authors:
#         co_author_avg_citations += author_data[co_author]['average_citations']
#         co_author_ids.append(assigned_ids[co_author])
#     each_author['co_authors'] = co_author_ids
#     each_author['co_author_avg_citations'] = co_author_avg_citations/len(co_author_ids) if len(co_author_ids) != 0 else 0
#     data_to_df.append(each_author)
    
# authorsData = pd.DataFrame.from_dict(data_to_df, orient='columns')

# authorsData['h_index'] = authorsData.apply(lambda x: get_h_index(x.citations), axis=1)



# #2. Computation of authors page rank

# ### AUTHOR PAGE RANK ###
# #Data Pre-processing: building the dataset on which the author network will be built
# train = authorsData.copy().drop(columns=['num_citations', 'h_index','paper_count', 'citations']).dropna(axis = 0, subset=['co_authors'])
# train = train[train.co_authors != '[]']
# train['author_id'] = pd.to_numeric(train['author_id'])

# # Building up the network to compute author page rank: 
# G = nx.Graph()
# for i in tqdm(range(len(train))):
#     auth = train.iloc[i]['author_id']
#     for neighbor in train.iloc[i]['co_authors']  :
#         if G.has_edge(auth, neighbor):
#             G.add_edge(auth, neighbor, weight = G[auth][neighbor]['weight']+1)
#         else:
#             G.add_edge(auth, neighbor, weight = 1)
            
# score_authors = nx.pagerank(G, alpha=0.55, max_iter=100, tol=1.0e-6, nstart=None, weight='weight', dangling=None)

# #Saving the page rank by author id
# authorPRK = pd.DataFrame.from_dict(score_authors, orient = "index")
# authorPRK["author_id"] = authorPRK.index
# authorPRK.columns = ["pagerank_author", "author_id"]
# authorPRK.to_csv("pagerank_author.csv",index = False)


# #3. Computation of publication page rank

# # Building up the network to compute the pagerank for publication
# G1 = nx.Graph()
# for i in range(len(datasetForScoring)):
# # for i in range(100): #Only on a sample
#     G1.add_node(datasetForScoring['refid'][i])
#     auth = datasetForScoring['refid'][i]
    
#     for e in list(str(datasetForScoring["references"][i]).lstrip("[").rstrip("]").replace(" ","").split(",")):
#         try:
#             if G1.has_edge(auth, e):
#                 G1.add_edge(auth, e, weight = G[auth][e]['weight']+1)
#             else:
#                 G1.add_edge(auth, e, weight = 1)
#         except:
#             continue
        
# score_publication = nx.pagerank(G1, alpha=0.85, tol=1.0e-6, nstart=None, weight=1, dangling=None)

# #Saving the page rank by paper id
# publiPRK = pd.DataFrame.from_dict(score_publication, orient = "index")
# publiPRK["publication_id"] = publiPRK.index
# publiPRK.columns = ["pageRankPublication", "publication_id"]
# publiPRK["publication_id"] = publiPRK["publication_id"].str.replace("'","")
# publiPRK = publiPRK.reset_index(drop = True)

# publiPRK.to_csv("pagerank_publication.csv",index = False)

# #Integration of the variable Page Rank for publication datasetForScoring
# enhancedDatasetForScoring = pd.merge(datasetForScoring,publiPRK, left_on = "refid", right_on = "publication_id", how = "left").drop(columns= ["publication_id"])
# enhancedDatasetForScoring = enhancedDatasetForScoring.drop_duplicates(subset='refid', keep="last") #Temporary patch to manage the case where twice Page rank for some publications



# #4. Computation of Author Scoring

# #Dataset to consolidate Author Page Rank and Publication Rank in a way to compute authorP2
# dfAuthorP2 = pd.merge(authorsData[["author_id","name"]],authorPRK, on = "author_id", how = "left").reset_index(drop=True)
# dfAuthorP2["name"] = dfAuthorP2["name"].str.replace("_"," ")

# # Extract enhancedDatasetForScoring "paper_refid" &"paper_authors"
# authorsfromDf = enhancedDatasetForScoring[["refid","authors"]].reset_index(drop = True)
# # authorsfromDf = authorsfromDf[(authorsfromDf["authors"].isna() == False)]
# authorsfromDf = pd.DataFrame(authorsfromDf.authors.tolist(), index = authorsfromDf.refid).stack().reset_index(level=1, drop=True).reset_index(name='authors')[['authors','refid']]

# #Computing the sum of publication page rank for each paper
# dfAuthorP2withPRPubli = pd.merge(authorsfromDf,publiPRK, left_on = "refid", right_on = "publication_id", how = "left").drop(columns = ["refid", "publication_id"]).groupby("authors").sum()
# dfAuthorP2withPRPubli["authors"] = dfAuthorP2withPRPubli.index #Reformatting
# dfAuthorP2withPRPubli = dfAuthorP2withPRPubli.reset_index(drop=True)

# dfAuthorP2Final = pd.merge(dfAuthorP2,dfAuthorP2withPRPubli, left_on = "name", right_on = "authors", how = "left").drop(columns = "name")

# # ######### Author Scoring #########
# dfAuthorP2Final["pagerank_author_norm"] = (dfAuthorP2Final["pagerank_author"]-dfAuthorP2Final["pagerank_author"].mean())/dfAuthorP2Final["pagerank_author"].std()
# dfAuthorP2Final["pagerank_publication_norm"] = (dfAuthorP2Final["pageRankPublication"]-dfAuthorP2Final["pageRankPublication"].mean())/dfAuthorP2Final["pageRankPublication"].std()

# dfAuthorP2Final["authorP2"] = 0.25*dfAuthorP2Final["pagerank_author_norm"] + 0.75*dfAuthorP2Final["pagerank_publication_norm"]



# #1. Influence score consolidation

#     # Consolidate Author Score for each paper
# authorP2Data = dfAuthorP2Final[["authors","authorP2"]]
# # enhancedDatasetForScoring = enhancedDatasetForScoring[(enhancedDatasetForScoring["authors"].isna() == False)]
# authorToPaper = pd.DataFrame(enhancedDatasetForScoring[["refid","authors"]].authors.tolist(), index=enhancedDatasetForScoring[["refid","authors"]].refid).stack().reset_index(level=1, drop=True).reset_index(name='authors')[['authors','refid']]

# authorP2Conso = pd.merge(authorToPaper,authorP2Data, on = "authors", how = "left")

# # Consolidate AuthorP2 for each paper as followed: 0.5 * Max page rank + 0.5 * average of the page rank of all the authors
# maxAuthorScore = authorP2Conso.groupby('refid').agg({'authorP2': 'max'})
# meanAuthorScore = authorP2Conso.groupby('refid').agg({'authorP2': 'mean'})

# authorScoring = pd.merge(maxAuthorScore,meanAuthorScore, on = "refid", how = "inner").rename(columns = {"authorP2_x" : "maxAuthorScore","authorP2_y" : "meanAuthorScore"})
# authorScoring["refid"] = authorScoring.index
# authorScoring = authorScoring.reset_index(drop = True)

# authorScoring["authorP2"] = 0.5*authorScoring["maxAuthorScore"] + 0.5*authorScoring["meanAuthorScore"]
# authorScoring = authorScoring.drop(columns = ["maxAuthorScore","meanAuthorScore"])

# #Integration of the variable authorP2 for datasetForScoring
# DatasetReadyForScoring = pd.merge(enhancedDatasetForScoring,authorScoring, on = "refid", how = "left")

# # Influence Score Computation Dataset Overview
# DatasetReadyForScoring.head()










myPath = 'C:/projects/machineLearning/hecCourse/metadataWithQuotationLabels.csv'
datasetForScoring.to_csv(myPath, encoding='utf-8', index=False)

#TODO THE DISTRIBUTION OF THE NB QUOTATIONS AND YEAR
# TODO: Transform the other categorical variables in one hot encoding if possible

print("Number of Papers in the CORD-19 dataset :",dfPaperList.shape[0])

print("Number of Citations found in the CORD-19 dataset :",dfCitationsFlat.shape[0])

print("Citations with no title: ",sum(1 if x == "" else 0 for x in dfCitationsFlat["title"]))

#How many duplicates? 
print("Number of duplicated research paper titles: ",len(dfPaperList["title"])-len(dfPaperList["title"].drop_duplicates()))

print("Number of duplicated citations titles: ",len(dfCitationsFlat["title"])-len(dfCitationsFlat["title"].drop_duplicates()))

#Dataframe Visualization
print("Number of Papers that will be scored: ", datasetForScoring.shape[0])
datasetForScoring.head()



"""===================================================
Etape 1 : Preprocess
==================================================="""

# Identifier articles sans abstract et les retirer
NAs = pd.isna(raw_metadata['abstract']) 
metadata = raw_metadata[np.invert(NAs)]
print(len(metadata))

# Reset les index metadata
metadata = metadata.reset_index(drop=True)
# important to ensure the articles are in March 2020



def selectRandomArticles(num, data):
    articleListIndex = []
    for _ in range(0, num):
        articleListIndex.append(random.randint(1,len(data)))
    return articleListIndex

idx = selectRandomArticles(500, metadata)

# Extract ID and abstract from dataset
df_abstracts = metadata.loc[idx ,['cord_uid', 'abstract']]  #Nombre d'abstranct sélectionnés


def remove_accented_chars(text):
    """remove accented characters from text, e.g. café"""
    text = unidecode.unidecode(text)
    return text


df_abstracts['abstract'] = df_abstracts.apply(lambda x: remove_accented_chars(x['abstract']), axis=1) # convert the accents into non accents in each

# Créer une liste des abstract nettoyé des chiffre et caractères spéciaux
list_abs = []
for i in df_abstracts['abstract']:

    list_abs.append(re.sub(r'[^a-zA-Z]+',' ',i)) # !!! Modifier pour ne pas retirer les accents
    #list_abs.append(i)
print("Nombre d'article retenus: ",len(list_abs))
 

print(list_abs[0])
print(metadata['abstract'][0])


#-----------------------------------------------------------------------
#
#
#         FUNCTION TO RETURN 30 ARTICLES FOR LABELLING
#
#
#-----------------------------------------------------------------------

myPath = 'C:/projects/machineLearning/hecCourse/'
def returnArticlesToRead(myPath, toCsv = True, numGab=20, numSimon=20, numFranck=10):
    randomArticlesList = []
    chunkOfArticles = []
    numberOfArticlesCorpus = metadata['abstract'].shape[0]
    listOfLabeller = [['Gabriel', numGab], ['Simon', numSimon], ['Franck', numFranck]]
    randomArticlesDict = {}
    # for reproducibility of the articles to retrieve
    random.seed(10)
    # generate a list of list of the articles randomly chosen for labelling
    for labeller in listOfLabeller:
        for _ in range(0, labeller[1]):
            articleIdx = random.randint(1,numberOfArticlesCorpus)
            articleIdentifier = metadata.loc[articleIdx,'cord_uid']
            articleAbstract = metadata.loc[metadata['cord_uid'] == articleIdentifier ,'abstract'].values[0]
            chunkOfArticles.append([articleIdentifier, articleAbstract,'','','','',''])

            if len(chunkOfArticles) == labeller[1]:
                randomArticlesList.append(chunkOfArticles)
                chunkOfArticles = []

        for idx, listOfArticlesId in enumerate(randomArticlesList):
            randomArticlesDict[listOfLabeller[idx][0]] = listOfArticlesId

    if toCsv:
        for idx, labeller in enumerate(listOfLabeller): 
            print(labeller)   
            df = pd.DataFrame(randomArticlesDict[labeller[0]], columns =['cord_uid','Abstract', 'topic 1', 'topic 2', 'topic 3', 'topic 4', 'topic 5'])    
            df['Labeller'] = labeller[0]

            if idx == 0:
                articlesToReadDf = df
            else:
                articlesToReadDf = pd.concat([articlesToReadDf, df])
        articlesToReadDf.to_csv(myPath+'ArticlesToRead.csv', encoding='utf-8', index=False)

    return randomArticlesDict
    
# to generate the list

dictionaryOfArticles = returnArticlesToRead(myPath, numGab=20, numSimon=20, numFranck=10)
articlesGab = dictionaryOfArticles['Gabriel'] # list of list de [article id, abstract]
articlesSimon = dictionaryOfArticles['Simon'] # list of list de [article id, abstract]
articlesFranck = dictionaryOfArticles['Franck'] # list of list de [article id, abstract]





"""=========================================================
Etape 2 : Création matrice de fréquence et matrice TF-IDF
========================================================="""

# =============================================================================
# # document pour pratique 
# list_abs = ["This is THE first sentense, it's a test checking 4 the code!",
#             "This is the SECOND of all documents, trying to get all TYPE of characters"]
# =============================================================================

# Définir le stemmer :
stemmer = SnowballStemmer('english')
# Ajouter un stemmer à la fonction "CountVectorizer" :
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

#my_stopwords_list = ['if', 'then', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

# 1- Initialiser la fonction StemmedCountVectorizer 
scv = StemmedCountVectorizer(strip_accents = 'ascii'# Retirer les accents sur les lettres
                             ,lowercase = True      # Retirer les majuscules 
                             ,stop_words='english'  # Retirer les STOPWORDS à aprtir d'une liste
                             #,stop_words= my_stopwords_list  # Retirer les STOPWORDS à aprtir d'une liste
                             ,analyzer='word'
                             #,min_df=2  # Élimine les mots qui apparaissent dans moins de 2 documents
                             ,max_df=0.70  # Élimine les mots qui apparaissent dans plus de 75% des documents
                             )

# 2- Créer la matrice "count_matrix" avec le décompte de chaque mot par document
count_matrix = scv.fit_transform(list_abs)

# =============================================================================
# # (PEUT ÊTRE COMMENTÉ) Voir la liste des mots de la matrice
# scv.vocabulary_
# # Voir la liste des STOPSWORDS du modèle
# scv.stop_words_
# # Voir la shape de la matrice : N documents X M words
# count_matrix.shape
# 
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
Etape 3 : Clustering avec K-Means
========================================================="""

# Voir: https://stackoverflow.com/questions/50827222/understanding-and-applying-k-means-clustering-for-topic-modeling
from sklearn.cluster import KMeans

n_clust = 10
n_bestwords = 25
#start = time()
# Initiatialiser le K-means model
kmeans_model = KMeans(n_clusters=n_clust, random_state=0)
# Fitter le K-means model
kmeans_model.fit(tfidf_matrix)

# "Sort" de l'importance des mots par cluster 
ordered_terms_kmeans = kmeans_model.cluster_centers_.argsort()[:, ::-1]
terms_tfidf = scv.get_feature_names() # Extraire le label des features

# =============================================================================
# # Pour imprimer principaux mots
# print("Resultats K-Means clustering")
# for i in range(n_clust):
#     print("Cluster", i+1),
#     for position in ordered_terms_kmeans[i,0:n_bestwords]:
#         print(terms_tfidf[position]),
#     print ("\n")
# =============================================================================

#stop = time()
#print(stop-start)

# Array des n_bestword par cluster
all_list = []
for i in range(n_clust):
    temp_list = []
    for position in ordered_terms_kmeans[i,0:n_bestwords]:
        temp_list.append(terms_tfidf[position])
    all_list.append(temp_list)
col_name = []
for i in range(len(all_list)):
    col_name.append("Cluster "+str(i+1))
arr = np.array(all_list)
bestwords_by_cluster = pd.DataFrame(arr.T, columns = col_name)
print(bestwords_by_cluster)

# Analyse de la présence de même mots dans les clusters 
score_list = []
for list_i in all_list:
    score_temp = 0
    for list_j in all_list:
        if list_i != list_j:
            for word in list_i:
                if word in list_j:
                    score_temp += 1
    score_list.append(score_temp)

# Nombre de répétition en pourcentage
relativ_score_list = (np.array(score_list) / (n_clust*n_bestwords))*100 # !!! Modifier le dénominateur  
# Mettre le tout dans un dataframe
word_rep = pd.DataFrame([np.array(score_list),relativ_score_list], 
                        columns = col_name,
                        index = ['Number rep', '% of words'])
# Afficher le résultat
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(word_rep)

           

"""=========================================================
Etape 4 : Clustering avec LDA (Latent Dirichlet Allocation)
========================================================="""    
# Voir: https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py

from sklearn.decomposition import LatentDirichletAllocation


start = time()
lda_model = LatentDirichletAllocation(n_components=3, random_state=0, max_iter=5)
lda_model.fit_transform(count_matrix)

   
# Pour faire ressortir les termes les plus frequents dans les k clusters
ordered_terms_lda = lda_model.components_.argsort()[:, ::-1]
terms_count = scv.get_feature_names()


print("Resultats clustering LDA \n")
for i in range(3):
    print("Cluster", i+1),
    for position in ordered_terms_lda[i,0:10]:
        print(terms_count[position]),
    print ("\n")

stop = time()
print(stop-start)




"""=========================================================
TEST matrice TF-IDF
========================================================="""
# =============================================================================
# from sklearn.feature_extraction.text import TfidfVectorizer 
# 
# # Document de test 
# docs = ["apple apple apple apple orange orange orange peach banana",
#        "orange orange orange peach peach pear pineapple prune", 
#        "prune prune prune prune orange orange orange cherry",
#        "kiwi orange orange orange tangerine grapefruit lime lime lemon",
#        "kiwi kiwi orange orange orange strawberry raspberry pear cherry"
#        ]
# print(docs)
# len(docs)
# 
# # Initialiser le TfidfVectorizer :  
# tfidf_vectorizer=TfidfVectorizer(use_idf=True) 
# 
# # Passer la liste de documents dans la fonction "fit_transform"
# tfidf_matrix = tfidf_vectorizer.fit_transform(docs)
# 
# # place tf-idf values in a pandas data frame 
# df = pd.DataFrame(tfidf_matrix.T.todense(), 
#                   index=tfidf_vectorizer.get_feature_names(), 
#                   columns=["doc1", "doc2", "doc3", "doc4", "doc5"]) 
# # Print dataframe
# print(df)
# 
# =============================================================================




