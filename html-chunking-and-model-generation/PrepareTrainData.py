#!/usr/bin/env python
# coding: utf-8

# In[119]:


'''importing libraries'''
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('stopwords')
import string
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import re
import unicodedata
import numpy as np
import ast
import math
import datetime
import sys

class PrepareTrainData:
    
    ngram = 2
    similarityMethod = 'cosine'
    #similarityMethod = 'jaccard'

    
    
    def _init_(self):
        self.getChunksWithSimilarityValue()
        self.getChunksWithHeighestSimilarityValue()
        self.prepareGroundTruthData()
        print('here')

    '''---------------------------------------START OF GENERIC METHODS---------------------------------------------'''
    '''method for cleaning string'''
    def clean_string(self,data): 

        custom_string_punctuation = re.sub(r"[$-:.%,£_]", "", string.punctuation) #removing some element from default string punctuation

        try:
            data = unicodedata.normalize("NFKD",data)
            data = data.strip()
            data = re.sub(r'[\n\r\t]','', data)
            data = data.lower()
            data = re.sub("([^-_a-zA-Z0-9!@#%&=,'\";:~`č₽€£\$\^\(\)\+\[\]\.\{\}\|\?\<\>\\]+|[^\s]+)",'', data)
            data = re.sub(r'<[^>]+>','', data) #removing html tags
            data = re.sub(r'a€™|a€tm|a€¦|a€|a€“|a€¢|â€™|â€tm|â€¦|â€|â€“|â€¢|i€', '', data)
            data = re.sub(r'\^[a-zA-Z]\s+', '', data) # Remove single characters from the start
            data = re.sub(r'\s+', ' ', data, flags=re.I)# Substituting multiple spaces with single space
            data = re.sub(r'^b\s+', '', data) # Removing prefixed 'b'
            data = data.replace('&amp;','and')
            data = data.replace('atm','') #for removing atm from all masteratms or membersatm string
            data = re.sub(r"([a-zA-Z])(\d+)", r"\g<1> \g<2>", data) # a34301 => 34301
            data  = re.sub(r"\b[a-zA-Z]\b", "", data) #removing single caracter
            data = re.sub(r"(\d+)(k)", r"\g<1>000", data)
            data = re.sub('(?<=\d),(?=\d{3})', '', data) #6,576,570 => 6576570
            #data = re.sub('(?<=\d) (?=\d)', ' ', data)
            #data = re.sub('(?<=[a-zA-Z]),(?=[a-zA-Z\s])', ' ', data)
            #data = re.sub('(?<=\d) (?=\d{3})', '', data)
            #cleaned_chunk = re.sub('(?<=\d) (?=\d)', '.', cleaned_chunk)
            #data = re.sub('(?<=\d) (?=\d)', '-', data)
            #data = re.sub(r"(\d+)-(00)", r"\g<1>", data) # e.g 34567-00 => 34567
            data = data.replace(' euro ',' € ')
            data = data.replace(' eur ',' € ')
            data = data.replace(' dollar ',' $ ')
            data = data.replace(' usd ',' $ ')
            
            data = re.sub(r"(\d+).(\d+) (€)|(\d+).(\d+)(€)|(\d+) (€)|(\d+)(€)", r"€ \g<1>", data) # 2984 € => € 2984
            data = re.sub(r"(\d+).(\d+) (\$)|(\d+).(\d+)(\$)|(\d+) (\$)|(\d+)(\$)", r"$ \g<1>", data) # 2984 $ => $ 2984
            data = re.sub(r"(\d+).(\d+) (£)|(\d+).(\d+)(£)|(\d+) (£)|(\d+)(£)", r"£ \g<1>", data) # 2984 £ => £ 2984

            data = re.sub('(?<=\d{4})-(?=\d{4})', ' to ', data) #1000-2000=> 1000 to 20000 for NER product to cardinal
            data = re.sub(r"["+custom_string_punctuation+"]", "", data) #removing some string puntuations ['!"#;<=>?@[\\]^_`{|}~']
            #data = word_tokenize(data)
            #print(data)
            #data = ' '.join([word for word in data if word not in custom_string_punctuation])
            #data = ' '.join([word for word in data.split() if word not in stopwords])
            # Lemmatization
            data = data.split()
            data = [WordNetLemmatizer().lemmatize(word) for word in data]
            data = ' '.join(data)
            #while re.match(r'^[a-zA-Z0-9]\s+', document):
                #document = re.sub(r'^[a-zA-Z0-9]\s+', '', document)
            
            return data
        
        except Exception as e:
            
            print('clean string failure !!!', data, e)
            return ''
        
    '''---------------------------------------END OF GENERIC METHODS---------------------------------------------'''
    
    
    
    '''----------------------START OF METHODS FOR SIMILARITY MEASURE---------------------------------------------'''
    '''method for getting cosine similarity of two vectors'''
    def cosine_sim_vectors(self,vec1,vec2):
        vec1 = vec1.reshape(1,-1)
        vec2 = vec2.reshape(1,-1)
        return cosine_similarity(vec1,vec2)[0][0]

    '''method for getting cosine similarity of two strings'''
    def get_cosine_similarity(self,str1,str2):
        
        cleaned = list(map(self.clean_string,[str1,str2]))
        #bigram_vectorizer = CountVectorizer(ngram_range=(1, self.ngram),token_pattern=r'\b\w+\b', min_df=1)#bigram counter vectorization
        bigram_vectorizer = CountVectorizer(ngram_range=(1, self.ngram),token_pattern=r'\b\w+\b',stop_words='english')#bigram counter vectorization

        vectorizer = bigram_vectorizer.fit_transform(cleaned)
        #vectorizer = CountVectorizer().fit_transform(cleaned)
        vectors = vectorizer.toarray()
        return self.cosine_sim_vectors(vectors[0], vectors[1])

    '''method for getting similarity'''
    def checkSimilarity(self,str1, str2):
        
        if self.similarityMethod == 'cosine':
            similarity = self.get_cosine_similarity(str1, str2)
        elif self.similarityMethod == 'jaccard':
            similarity = self.get_jaccard_sim(str1, str2)
            
        return similarity
    
    '''method for calculate jaccard similarity'''
    def get_jaccard_sim_old(self,str1, str2):
        a = set(prp.clean_string(str1)) 
        b = set(prp.clean_string(str2))
        intersection = a.intersection(b)
        union = a.union(b)
        return len(intersection) / len(union)
    
    def get_jaccard_sim(self,str1, str2):
        cleaned = list(map(self.clean_string,[str1,str2]))
        #print(cleaned)
        bigram_vectorizer = CountVectorizer(ngram_range=(1, self.ngram),token_pattern=r'\b\w+\b',stop_words='english')#bigram counter vectorization
        vectorizer = bigram_vectorizer.fit_transform(cleaned)
        vectors = vectorizer.toarray()
        #return jaccard_score(vectors[0], vectors[1],average='macro')
        #return jaccard_score(vectors[0], vectors[1],average='micro')
        return jaccard_score(vectors[0], vectors[1],average='weighted')


        
        
    
    '''--------------------------------END OF METHODS FOR SIMILARITY MEASURE--------------------------------------'''
    
    
    
    '''-------------------------START OF METHODS FOR GENERATING TRAIN DATA----------------------------------------'''
    '''method for get all chunks with similarity value'''
    def getChunksWithSimilarityValue(self):
        
        '''timetracking start'''
        timeStart = datetime.datetime.now()

        '''improting annotated dataset'''
        annotatedDf = pd.read_csv('../data/jobs-europe/annotated_JobsEurope-opt.csv',sep=';')
        
        '''importing extructed html text dataset'''
        df = pd.read_csv('../data/parsed-data/jobs-europe-chunks-same-to-annotated.csv')

        '''we consider only 17 columns from all columns'''
        #escapeColumns = ['id','user_id','title','url','date_posted','valid_through','estimated_salary','job_category','industry','salary_currency','same_as']
        escapeColumns = ['id','user_id','title','url','date_posted','valid_through','estimated_salary','job_category','industry','salary_currency','same_as','incentive_compensation','special_commitments','job_benefits']
        allColumns = list(annotatedDf.columns)
        acceptedColumns = [ele for ele in allColumns if ele not in escapeColumns]

        '''functionality for calculating chunkwise similarity with annotated dataset'''
        labeledChunks = []
        for index, row in df.iterrows():
            try:
                datapointId = row['data_point_id']
                chunkId = row['chunk_id']
                chunk = self.clean_string(row['chunk'])
                
                #if(len(chunk.split()) < 2 ):continue #discard chunks with one word

                annotatedRow = annotatedDf.loc[annotatedDf['id'] == datapointId]

                '''iteration over annotated jobs europe columns'''
                for columnName in acceptedColumns:
                    columnValue = annotatedRow[columnName].values[0]
                    
                    if isinstance(columnValue, float) and math.isnan(float(columnValue)): continue #avoiding NAN values
                    columnValue = self.clean_string(columnValue) #preprocess
                    similarity = self.checkSimilarity(chunk, columnValue) #similarity checking            
                    labeledChunks.append([datapointId,row['chunk_id'],chunk,columnValue,columnName,similarity,row['non_translated_chunk']])

                print('getChunksWithSimilarityValue success !!!: ','index',index,'datapointId',datapointId,'chunkId',chunkId)
            
            except Exception as e:
                
                print('getChunksWithSimilarityValue failure !!!: ','index',index,'datapointId',datapointId,'chunkId',chunkId,'error',e)
                continue

        '''saving labeled chunks'''
        df = pd.DataFrame(labeledChunks)
        df.columns = ['data_point_id','chunk_id','chunk','compared_with','labeled_category','similarity_threshold','non_translated_chunk']
        df.to_csv(r'../data/labeled-data/'+self.similarityMethod+'/'+str(self.ngram)+'-gram-sim.csv', index = False)

        print('getChunksWithSimilarityValue time Required:',datetime.datetime.now()-timeStart) #time tracking end
        
        return df
    
    
    '''methgod for getting chunks with heighest similarity value among all chunks'''
    def getChunksWithHeighestSimilarityValue(self):
        '''importing bigram cosine similarity datset'''
        df = pd.read_csv('../data/labeled-data/'+self.similarityMethod+'/'+str(self.ngram)+'-gram-sim.csv')

        '''functionality for prparing chunkwise similarity dataset'''
        df = df[df['similarity_threshold'] > 0]
        df.reset_index(drop=True,inplace = True)
        
        '''collecting all chunk id'''
        chunkIds = df['chunk_id'].unique()
        
        '''iteration over chunk ids for getting chunkwise heighest similarity'''
        allChunkDf = pd.DataFrame(columns = df.columns)
        for chunkId in chunkIds:
            
            try:
                indvChunkDf = df.copy()
                '''collecting chunkwise heighest similarity'''
                indvChunkDf.query("chunk_id == " +str(chunkId), inplace = True)
                indvChunkDf.query("similarity_threshold == similarity_threshold.max()", inplace = True)

                '''if chunkwise heighest similarity exists for multiple instance take first one drop others'''
                if len(indvChunkDf) > 1:
                    indvChunkDf.sort_values("similarity_threshold", inplace = True) 
                    indvChunkDf.drop_duplicates(subset='similarity_threshold', keep='first',inplace = True)

                indvChunkDf.reset_index(drop = True,inplace = True)
                indvChunkDf = indvChunkDf.head(1); #keeping first instance for each unique chunk id with the heighest similarity value
                allChunkDf = allChunkDf.append(indvChunkDf, ignore_index=True)
                
                print('getChunksWithHeighestSimilarityValue success !!!: ','chunkId',chunkId)
            
            except Exception as e:
                
                print('getChunksWithHeighestSimilarityValue failure !!!: ','chunkId',chunkId,'error',e)
                continue
                
        allChunkDf.loc[(allChunkDf.labeled_category =='qualifications'),'labeled_category']='skills'
        allChunkDf.to_csv(r"../data/labeled-data/"+self.similarityMethod+"/"+str(self.ngram)+"-gram-sim-max.csv", index = False)
        
        return allChunkDf
    
    def prepareGroundTruthData(self):
        
        df = pd.read_csv("../data/labeled-data/"+self.similarityMethod+"/"+str(self.ngram)+"-gram-sim-max.csv")
        
        for i in [0.3,0.35,0.4]:
            c_df = df[df['similarity_threshold'] >= i]
            c_df.reset_index(drop=True,inplace = True)
            c_df.to_csv(r"../data/labeled-data/ground-truth/"+str(self.ngram)+"-gram-"+self.similarityMethod+"-"+str(i)+"-gt.csv", index = False)


        
    '''----------------------------END OF METHODS FOR GENERATING GROUND TRUTH DATA----------------------------------------'''
        
PrepareTrainData()._init_()

