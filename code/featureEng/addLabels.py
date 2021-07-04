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
metadata = pd.read_csv("C:/projects/machineLearning/hecCourse/metadata_lang.csv")

metadata = metadata[["cord_uid","sha", "title", "abstract", "publish_time"]].rename(columns = {"sha" : "paper_id"})
dfPaperList = pd.merge(metadata, authorsDf, on = "paper_id", how = "left")

dfPaperList["year"] = 0
dfPaperList["refid"] = ""

for i in tqdm(range(len(dfPaperList))):
    # TODO: include explanation
    dfPaperList["refid"][i] =  hashlib.sha1(str(dfPaperList["title"][i]).lower().encode()).hexdigest()  
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

datasetForScoring.to_csv('C:/projects/machineLearning/hecCourse/metadataWithQuotationLabels.csv', encoding='utf-8', index=False)


