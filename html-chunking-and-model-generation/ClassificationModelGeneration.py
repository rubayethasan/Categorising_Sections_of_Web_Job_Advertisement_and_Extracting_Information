#!/usr/bin/env python
# coding: utf-8

# In[7]:


'''importing libraries'''
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()
import imblearn
from imblearn.over_sampling import RandomOverSampler
from operator import itemgetter
import pickle
import sys

class ClassificationModelGeneration:
    
    ngram = 2
    similarityThreshold = 0.35
    similarityMethod = 'cosine'
    #similarityMethod = 'jaccard'

    trainDataFile = str(ngram)+'-gram-'+similarityMethod+'-'+str(similarityThreshold)+'-gt'
    savedModelFile = str(ngram)+"-gram-"+str(similarityThreshold)+"-similarity-LG"
    #savedModelFile = str(ngram)+"-gram-"+str(similarityThreshold)+"-similarity-SVC"

    nerPosFeaturesFile = str(ngram)+"-gram-"+str(similarityThreshold)+"-similarity-ner-pos-features-name"
    nGramFeaturesFile = str(ngram)+"-gram-"+str(similarityThreshold)+"-similarity-ngram-features-name"
    allFeaturesFile = str(ngram)+"-gram-"+str(similarityThreshold)+"-similarity-all-features-name"
    posFeaturesFile = str(ngram)+"-gram-"+similarityMethod+'-sim-'+str(similarityThreshold)+"-pos"
    nerFeaturesFile = str(ngram)+"-gram-"+similarityMethod+'-sim-'+str(similarityThreshold)+"-ner"

    def _init_(self):
        
        '''importing training data'''
        df = pd.read_csv("../data/labeled-data/ground-truth/"+self.trainDataFile+".csv")
        df.drop(df.loc[df['labeled_category']=='job_fields'].index, inplace=True)
        df.reset_index(drop=True,inplace = True)

        #df = df[ df['category'].isin(['skills','experience_requirements','base_salary','education_requirements','qualifications'])]
        #df = df.reset_index(drop=True)
            
        
        '''feature extraction form dataset'''
        features = self.featureExtruction(df)

        '''generating labels'''
        labels = self.generatingLabels(df)
        
        print('feature extraction and label generation done')

        '''classification over the data'''
        self.classify(features,labels)
        

    '''method for merging auxilary features'''    
    def mergingAuxilaryFeatures(self,df):

        '''creating number token and word number feature'''
        df['num_words'] = df['chunk'].apply(lambda x: (len(x.split()) - len([ x for x in x.split() if x.isdigit() ])) / len(x.split())  )
        df['number_tokens'] = df['chunk'].apply(lambda x : len( [ x for x in x.split() if x.isdigit() ]) / len(x.split()) )

        
        
        return df


    '''method for creating n grams'''
    def creatingNgramFeatures(self,df):

        '''initiating vectorizer for n gram feature'''
        #vectorizer = CountVectorizer(ngram_range=(1, self.ngram),binary=True,token_pattern=r'\b[^\d\W]+\b', min_df=2,stop_words='english')
        vectorizer = CountVectorizer(ngram_range=(1, self.ngram),token_pattern=r'\b[^\d\W]+\b', min_df=2)


        '''n gram feature creation for chunks'''
        ngrams = vectorizer.fit_transform(df.chunk).toarray()
        ngrams = pd.DataFrame(ngrams)
        ngrams.columns = vectorizer.get_feature_names()
        
        #ngrams = ngrams.div(ngrams.sum(axis=0), axis=1)
        #ngrams.drop([col for col, val in ngrams.sum().iteritems() if val <= 0.2], axis=1, inplace=True)


        return ngrams,vectorizer.get_feature_names()


    '''method for saving n gram feature names for further using during framing ngram features of un categorized data'''    

    def saveFeaturesName(self,featureNames,featureType):
        featureNames = pd.DataFrame(featureNames,columns=['feature-name'])
        featureNames.to_csv(r'../data/features/'+featureType+'.csv', index = False)

    '''method for generating named entities and pos tag by number of occourance frequency'''
    def generate_features(self,data):
        
        features_pos = defaultdict(list)
        features_ner = defaultdict(list)
        features_emb = []

        for i, row in data.iterrows():
            
            try:
                doc = nlp(row['chunk'])

                pos_tags_ = [token.tag_ for token in doc]
                ner_ = [ent.label_ for ent in doc.ents]

                # compute the POS tag distribution
                pos_tag_counter = Counter(pos_tags_)
                #tag_dist = {tag: pos_tag_counter[tag] / float(len(pos_tags_)) for tag in pos_tag_counter}
                tag_dist = {tag: pos_tag_counter[tag] for tag in pos_tag_counter}


                # compute the NER tag distribution
                ner_counter = Counter(ner_)
                #ner_dist = {ner_tag: ner_counter[ner_tag] / float(len(ner_)) for ner_tag in ner_counter}
                ner_dist = {ner_tag: ner_counter[ner_tag] for ner_tag in ner_counter}

                # add the features
                for tag_ in nlp.pipe_labels['tagger']:
                    features_pos[tag_].append(tag_dist[tag_] if tag_ in tag_dist else 0.0)
                    #features_pos[tag_].append(1.0 if tag_ in tag_dist else 0.0)

                for tag_ in nlp.pipe_labels['ner']:
                    features_ner[tag_].append(ner_dist[tag_] if tag_ in ner_dist else 0.0)
                    #features_pos[tag_].append(1.0 if tag_ in tag_dist else 0.0)
                    
                print('generate_features success !!! index', i )
                
            except Exception as e:
                
                print('generate_features failure !!! index', i, 'error', e )
                continue
                

        return features_pos, features_ner


    '''method for processing POS tag and Named Enitity features'''
    def processingNamedEntityAndPosTagFeatures(self,df):

        '''creating POS tag and Named Enitity features'''
        features_pos, features_ner = self.generate_features(df)
        
        df_pos = pd.DataFrame.from_dict(features_pos)
        df_ner = pd.DataFrame.from_dict(features_ner)

        '''avoiding the Named Entity and POS tag features which exist in very less instances'''
        df_pos.drop([col for col, val in df_pos.sum().iteritems() if val <= 0.2], axis=1, inplace=True)
        df_ner.drop([col for col, val in df_ner.sum().iteritems() if val <= 0.2], axis=1, inplace=True)
        
        #df_pos.drop([col for col, val in df_pos.sum().iteritems() if val <= 2], axis=1, inplace=True)
        #df_ner.drop([col for col, val in df_ner.sum().iteritems() if val <= 2], axis=1, inplace=True)

        return df_pos,df_ner

    '''method for extructing featres for classification'''    
    def featureExtruction(self,df):

        originalDfColumns = df.columns
        '''merging Named Entity and POS tag features with main dataset'''
        df_pos,df_ner = self.processingNamedEntityAndPosTagFeatures(df)
        df_pos.to_csv(r'../data/features/'+self.posFeaturesFile+'.csv', index = False)
        df_ner.to_csv(r'../data/features/'+self.nerFeaturesFile+'.csv', index = False)
        print('-----------ner and pos feature creation done---------------')
        
        '''saving pos and ner feature names for further use'''
        self.saveFeaturesName(list(df_pos.columns.values) + list(df_ner.columns.values),self.nerPosFeaturesFile)


        df = pd.concat([df, df_pos, df_ner],axis=1)

        '''merging Auxilary Features with Named Entity and POS tag features with main dataset'''
        df = self.mergingAuxilaryFeatures(df)

        '''combinning ngram features with other features'''
        ngrams, nGramFeaturesName = self.creatingNgramFeatures(df)
        
        print('-----------n gram feature creation done---------------')
        
        features = pd.concat([df, ngrams], axis=1)

        '''saving n grams and n gram feature names for further use'''
        ngrams.to_csv(r'../data/features/ngrams.csv', index = False)
        self.saveFeaturesName(nGramFeaturesName,self.nGramFeaturesFile)

        '''save all features for further analysis'''
        features.to_csv(r'../data/features/features_all.csv', index = False)
        '''removing basic columns from feature dataset'''
        #features = features.drop(features.columns[[0, 1, 2, 3, 4, 5, 6]], axis=1)
        features.drop(columns = originalDfColumns, axis=1, inplace = True)
        
        '''saving all gram feature names for further use'''
        self.saveFeaturesName(features.columns,self.allFeaturesFile)

        return features

    '''method for preparing the labels'''
    def generatingLabels(self,df):
        labels = [val_ for val_ in df.labeled_category.values]
        labels = pd.DataFrame(labels,columns=['labeled_category'])
        return labels

    '''method for classifiation'''
    def classify(self,features,labels):

        '''training test split'''
        X_train, X_test, y_train, y_test = train_test_split(features.to_numpy(), labels.to_numpy(), test_size=0.25, random_state=42)

        '''oversampling the traning set'''
        overSample = RandomOverSampler(random_state=42, sampling_strategy = 'auto')
        X_train, y_train = overSample.fit_sample(X_train, y_train)

        '''Initiating the classifier'''
        #model = LinearSVC(max_iter=100000)
        model = LogisticRegression(solver='lbfgs',multi_class='auto',max_iter=10000)

        '''model training'''
        model.fit(X_train, y_train)

        '''saving model for further use'''
        pickle.dump(model, open('../data/models/'+self.savedModelFile+'.sav', 'wb'))

        '''model prediction'''
        y_pred = model.predict(X_test)
        
        print('n-gram:',self.ngram,'sim-mth: ',self.similarityMethod,' sim-th:',self.similarityThreshold,' total_instance:',len(features),'  X_train:', len(X_train),'  y_train:', len(y_train),'  X_test:',len(X_test),'  y_test:',len(y_test))
        print(metrics.classification_report(y_test, y_pred))


ClassificationModelGeneration()._init_()


# In[ ]:




