#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''importing libraries'''
import multiprocessing
import os
from collections import Counter, defaultdict

import en_core_web_sm
import numpy as np
import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer

nlp = en_core_web_sm.load()
import pickle

from tqdm import tqdm


class ClassifyUsingSavedModel:
    
    
    ngram = 2
    similarityThreshold = 0.35
    similarityMethod = 'cosine'
    #similarityMethod = 'jaccard'

    savedModelFile = str(ngram)+"-gram-"+str(similarityThreshold)+"-similarity-LG"
    #savedModelFile = str(ngram)+"-gram-"+str(similarityThreshold)+"-similarity-SVC"

    posNerFeatureNames = str(ngram)+"-gram-"+str(similarityThreshold)+"-similarity-ner-pos-features-name"
    nGramFeaturesFile = str(ngram)+"-gram-"+str(similarityThreshold)+"-similarity-ngram-features-name"
    allFeaturesFile = str(ngram)+"-gram-"+str(similarityThreshold)+"-similarity-all-features-name"
    posFeaturesFile = str(ngram)+"-gram-"+similarityMethod+'-sim-'+str(similarityThreshold)+"-pos"
    nerFeaturesFile = str(ngram)+"-gram-"+similarityMethod+'-sim-'+str(similarityThreshold)+"-ner"
    
    
    jobPortal = 'euro-jobs'
    testDataFile = jobPortal + "-chunks"
    labeledDataFile = jobPortal + "-labeled-chunks"


    def _init_(self, display):

        '''importing training data'''
        df = pd.read_csv("../data/parsed-data/" + self.testDataFile + ".csv")
        # df = df.head(100000)

        display.set_state(6)

        '''feature extraction form dataset'''
        features = self.featureExtructionForClassification(df, display)
        #print('features ready')

        display.set_state(10)

        '''classification over the data'''
        detectedLabels = self.classifyUsingTrainedModel(features)
        #print('clssification done')

        labeledData = pd.concat([df, detectedLabels], axis=1)
        labeledData.to_csv(r'../data/labeled-data/' + self.labeledDataFile + '.csv', index=False)
        display.set_state(0)
        #print('labeled data saved')

    '''method for creating n grams'''

    def extractingNgramFeatures(self, df):

        ngramFeatures = pd.read_csv("../data/features/" + self.nGramFeaturesFile + ".csv")
        '''initiating vectorizer for n gram feature'''
        vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1,vocabulary=ngramFeatures['feature-name'])

        '''n gram feature creation for chunks'''
        ngrams = vectorizer.fit_transform(df.chunk).toarray()
        ngrams = pd.DataFrame(ngrams)
        ngrams.columns = vectorizer.get_feature_names()

        return ngrams

    '''method for generating named entities and pos tag by number of occourance frequency'''

    def generate_features_(self, data):

        features_pos = defaultdict(list)
        features_ner = defaultdict(list)
        features_emb = []

        # TODO parallelise

        with tqdm(total=len(data), unit=' chunks') as p_bar:
            for i, row in data.iterrows():

                try:
                    doc = nlp(row['chunk'])  # takes time

                    pos_tags_ = [token.tag_ for token in doc]
                    ner_ = [ent.label_ for ent in doc.ents]

                    # compute the POS tag distribution
                    pos_tag_counter = Counter(pos_tags_)
                    # tag_dist = {tag: pos_tag_counter[tag] / float(len(pos_tags_)) for tag in pos_tag_counter}
                    tag_dist = {tag: pos_tag_counter[tag] for tag in pos_tag_counter}

                    # compute the NER tag distribution
                    ner_counter = Counter(ner_)
                    # ner_dist = {ner_tag: ner_counter[ner_tag] / float(len(ner_)) for ner_tag in ner_counter}
                    ner_dist = {ner_tag: ner_counter[ner_tag] for ner_tag in ner_counter}

                    # add the features
                    for tag_ in nlp.pipe_labels['tagger']:
                        features_pos[tag_].append(tag_dist[tag_] if tag_ in tag_dist else 0.0)
                    for tag_ in nlp.pipe_labels['ner']:
                        features_ner[tag_].append(ner_dist[tag_] if tag_ in ner_dist else 0.0)

                    # print('generate_features success !!! index', i)
                    p_bar.update()

                except Exception as e:

                    print('generate_features failure !!! index', i, 'error', e)
                    continue

        return features_pos, features_ner

    def generate_features(self, data, display):

        features_pos = defaultdict(list)
        features_ner = defaultdict(list)
        features_emb = []

        # TODO parallelise
        self._chunk_index = data.columns.get_loc('chunk')

        # multiprocessing.set_start_method('spawn')

        features = []

        with multiprocessing.Pool(processes=os.cpu_count()) as p:
            with tqdm(total=len(data), unit=' chunks') as p_bar:
                for i, res in enumerate(p.imap(self.parallel_feature_gen, data.values.tolist())):
                    p_bar.update()
                    features.append(res)
                    # features_pos.append(res)

        display.set_state(8)

        features_pos = {k: [features[i][0][k] for i in range(len(features))] for k in features[0][0]}
        features_ner = {k: [features[i][1][k] for i in range(len(features))] for k in features[0][1]}

        return features_pos, features_ner

    def parallel_feature_gen(self, row):

        doc = nlp(row[self._chunk_index])  # takes time

        pos_tags_ = [token.tag_ for token in doc]
        ner_ = [ent.label_ for ent in doc.ents]

        # compute the POS tag distribution
        pos_tag_counter = Counter(pos_tags_)
        # tag_dist = {tag: pos_tag_counter[tag] / float(len(pos_tags_)) for tag in pos_tag_counter}
        tag_dist = {tag: pos_tag_counter[tag] for tag in pos_tag_counter}
        tag_dist = {tag: pos_tag_counter[tag] for tag in pos_tag_counter}

        # compute the NER tag distribution
        ner_counter = Counter(ner_)
        # ner_dist = {ner_tag: ner_counter[ner_tag] / float(len(ner_)) for ner_tag in ner_counter}
        ner_dist = {ner_tag: ner_counter[ner_tag] for ner_tag in ner_counter}

        features_pos = {}
        features_ner = {}

        # add the features
        for tag_ in nlp.pipe_labels['tagger']:
            features_pos[tag_] = tag_dist[tag_] if tag_ in tag_dist else 0.0
        for tag_ in nlp.pipe_labels['ner']:
            features_ner[tag_] = ner_dist[tag_] if tag_ in ner_dist else 0.0

        return features_pos, features_ner

    '''method for merging auxilary features'''

    def mergingAuxilaryFeatures(self, df):

        '''creating number token and word number feature'''
        df['num_words'] = df['chunk'].apply(
            lambda x: (len(x.split()) - len([x for x in x.split() if x.isdigit()])) / len(x.split()))
        df['number_tokens'] = df['chunk'].apply(lambda x: len([x for x in x.split() if x.isdigit()]) / len(x.split()))

        return df

    '''method for processing POS tag and Named Enitity features'''

    def processingNamedEntityAndPosTagFeatures(self, df, display):

        '''creating POS tag and Named Enitity features'''
        features_pos, features_ner = self.generate_features(df, display)

        df_pos = pd.DataFrame.from_dict(features_pos)
        df_ner = pd.DataFrame.from_dict(features_ner)

        '''avoiding the Named Entity and POS tag features which exist in very less instances'''
        df_pos.drop([col for col, val in df_pos.sum().iteritems() if val <= 0.2], axis=1, inplace=True)
        df_ner.drop([col for col, val in df_ner.sum().iteritems() if val <= 0.2], axis=1, inplace=True)

        return df_pos, df_ner

    '''method for mapping spacy POS and NER Features From Saved Data'''

    def mapPosNerFeaturesFromSavedData(self, pos_ner_features):

        saved_pos_ner_feature_names = pd.read_csv("../data/features/" + self.posNerFeatureNames + ".csv")
        saved_pos_ner_feature_names = saved_pos_ner_feature_names['feature-name'].values

        mapped_df = pd.DataFrame()
        existedColumns = []
        notExistedColumns = []

        for i in saved_pos_ner_feature_names:
            if i in pos_ner_features.columns:
                existedColumns.append(i)
            else:
                notExistedColumns.append(i)

        if len(existedColumns) > 0:
            mapped_df = pos_ner_features[existedColumns]

        for j in notExistedColumns:
            mapped_df[j] = 0  # TODO A value is trying to be set on a copy of a slice from a DataFrame.

        return mapped_df

    '''method for extructing featres for classification'''

    def featureExtructionForClassification(self, df, display):

        '''merging Named Entity and POS tag features with main dataset'''
        display.set_state(7)
        df_pos, df_ner = self.processingNamedEntityAndPosTagFeatures(df, display)
        # print('-----------ner and pos feature creation done---------------')

        display.set_state(9)

        '''mapping spacy POS and NER Features From Saved Data'''
        pos_ner_features = pd.concat([df_pos, df_ner], axis=1)
        pos_ner_features = self.mapPosNerFeaturesFromSavedData(pos_ner_features)
        df = pd.concat([df, pos_ner_features], axis=1)

        '''merging Auxilary Features with Named Entity and POS tag features with main dataset'''
        df = self.mergingAuxilaryFeatures(df)

        '''combinning ngram features with other features'''
        ngrams = self.extractingNgramFeatures(df)
        # print('-----------n gram feature extraction done---------------')

        df = df.drop(['data_point_id', 'chunk_id', 'chunk'], axis=1)

        # features = pd.concat([df, ngrams], axis=1)  # TODO find an alternative way to do this, too expensive for memory

        for (columnName, columnData) in df.iteritems():
            ngrams[columnName] = columnData

        # print(ngrams == features)

        # features = pd.merge(df, ngrams)

        '''dropping first three columns'''

        # print(len(ngrams.columns))

        return ngrams
        
        # print(len(features.columns))

        # return features

    '''method for classifiation'''

    def classifyUsingTrainedModel(self, features):
        '''importing previously build model'''

        model = pickle.load(open('../data/models/' + self.savedModelFile + '.sav', 'rb'))

        y_pred = []
        max_size = 10000
        start = 0

        with tqdm(total=len(features), unit=' chunks') as p_bar:
            while start < len(features):
                max_size = min(max_size, len(features) - start)
                a = features.iloc[start: start + max_size].to_numpy()
                start += max_size
                p_bar.update(max_size)

                y_pred.extend(model.predict(np.array(a)))

        return pd.DataFrame(y_pred, columns=['category'])
