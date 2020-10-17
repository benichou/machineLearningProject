# importing necessary libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
import time



#1
#read data set
meta_data = pd.read_csv('C:/projects/machineLearning/hecCourse/metadata.csv')
print('Original Size of Data:',meta_data.shape)

#drop rows with null values (based on abstract attribute)
meta_data.dropna(subset = ['abstract'],axis = 0, inplace = True)
print('Data Size after dropping rows with null values (based on abstract attribute):',meta_data.shape)



#handling duplicate data (based on 'sha','title' and 'abstract')
print(meta_data[meta_data.duplicated(subset=['sha','title','abstract'], keep=False) == True])
meta_data.drop_duplicates(subset=['sha','title','abstract'],keep ='last',inplace=True)
print('Data Size after dropping duplicated data (based on abstract attribute):',meta_data.shape)

#3
#function to deal with null values
#'No Information Available' will be replaced 
def dealing_with_null_values(dataset):
    dataset = dataset
    for i in dataset.columns:
        replace = []
        data  = dataset[i].isnull()
        count = 0
        for j,k in zip(data,dataset[i]):
            if (j==True):
                count = count+1
                replace.append('No Information Available')
            else:
                replace.append(k)
        print("Num of null values (",i,"):",count)
        dataset[i] = replace
    return dataset

meta_data = dealing_with_null_values(meta_data)





# pca to be put in a different file

from sklearn.decomposition import PCA
def pca_fun(n_components, data):
    pca = PCA(n_components=n_components).fit(data)
    data = pca.transform(data)
    return data



from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf(data):
    tfidf = TfidfVectorizer( stop_words='english',use_idf=True)
    tfidf_matrix = tfidf.fit_transform(data)
    return tfidf_matrix

tfidf_matrix = tfidf(meta_data['abstract'])

dir(tfidf_matrix)



# in order to explore which documents have more similar respresentaiton, consine simliartiy can be used
from sklearn.metrics.pairwise import linear_kernel
cosine_similarities = linear_kernel(tfidf_matrix[0:1], tfidf_matrix).flatten()

# 10 most related documents indices
related_docs_indices = cosine_similarities.argsort()[:-11:-1]
print("Related Document:",related_docs_indices)

# Cosine Similarties of related documents
print("Cosine Similarites of related documents",cosine_similarities[related_docs_indices])


meta_data.iloc[1]['abstract']


from wordcloud import WordCloud
import matplotlib.pyplot as plt


meta_data['index'] = meta_data.index
meta_data['index'] = 0 

allArticles = meta_data.loc[:, ['abstract','index']].groupby('index')['abstract'].apply(' '.join).reset_index()



wordcloud = WordCloud().generate(allArticles['abstract'][0])
plt.imshow(wordcloud, interpolation="bilinear")
plt.show()

from wordcloud import WordCloud
import matplotlib.pyplot as plt
wordcloud = WordCloud().generate(meta_data.iloc[0]['abstract'])
plt.imshow(wordcloud, interpolation="bilinear")






import gensim
from gensim.models import Doc2Vec

def doc2vec():
    document_tagged = []
    tagged_count = 0
    for _ in meta_data['abstract'].values:
        document_tagged.append(gensim.models.doc2vec.TaggedDocument(_,[tagged_count]))
        tagged_count +=1 
    d2v = Doc2Vec(document_tagged)
    d2v.train(document_tagged,epochs=d2v.epochs,total_examples=d2v.corpus_count)
    return d2v.docvecs.vectors_docs


start = time.time()

doc2vec = doc2vec()

end = time.time()

processingTime = end - start



import seaborn as sns
plt.figure(figsize=(16,16))
sns.heatmap(doc2vec,cmap="coolwarm")

plt.show()





# importing KMeans library of sklearn
from sklearn.cluster import KMeans

def kmeans(n_clusters):
    kmean_model = KMeans(n_clusters = n_clusters,random_state=0)
    return kmean_model


help(KMeans)

X = doc2vec



kmeans5 = KMeans(5)

km5 = kmeans5.fit_predict(X)
