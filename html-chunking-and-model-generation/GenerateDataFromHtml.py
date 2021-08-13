#!/usr/bin/env python
# coding: utf-8

# In[97]:


'''Importing libraries'''
import time
import pandas as pd
import numpy as np
import datetime
import cgi
import unicodedata
import re
import string
import ast
from collections import Counter, defaultdict
import spacy
import en_core_web_lg

nlp = en_core_web_lg.load()
import pickle
import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from html.parser import HTMLParser
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import geograpy
stopwords = stopwords.words('english')
from google_trans_new import google_translator

'''class for feeding html document to extract text data'''


class MyCustomHTMLParser(HTMLParser):
    restricted_strings_for_euro_jobs = ['location:', 'job category:', 'eu work permit required:', 'job reference:',
                                        'posted:', 'expiry date:', 'job description:', 'company info', 'phone:',
                                        'web site:', 'job views:', 'original source', 'save contact', 'company profile',
                                        'get job by email']
    prevNodeDataCheck = ['location:', 'job category:', 'eu work permit required:', 'job reference:', 'posted:',
                         'expiry date:', 'job description:', 'company info', 'phone:', 'web site:', 'job views:']
    chunkedData = []
    prevNodeData = ''
    origChunkedData = []

    # def handle_starttag(self, tag, attrs):
    # print("start tag:", tag)

    def handle_data(self, data):

        # print('original data: ',data )
        origData = data
        data = GenerateDataFromHtml().clean_string(data)
        words = data.split()
        # print("cleaned data:", data)
        if GenerateDataFromHtml().jobPortalName == 'euro-jobs':  # for parsing euro jobs data

            if data not in self.restricted_strings_for_euro_jobs:

                if len(words) > 0 and self.prevNodeData != '' and self.prevNodeData in self.prevNodeDataCheck:
                    combinedData = self.prevNodeData + ' ' + str(data)
                    if combinedData not in self.chunkedData:
                        self.chunkedData.append(combinedData)
                        self.origChunkedData.append(origData)
                        # print("recorded data  :", combinedData)

                elif len(words) > 1 and data not in self.chunkedData:
                    self.chunkedData.append(data)
                    self.origChunkedData.append(origData)
                    # print("recorded data  :", data)

        else:  # for parsing jobs-europe data
            if len(words) > 1 and data not in self.chunkedData:
                self.chunkedData.append(data)
                self.origChunkedData.append(origData)
                # print("recorded data  :", data)

        # print("original data  :",origData)
        self.prevNodeData = data

    # def handle_endtag(self, tag):
    # print("end tag :", tag)


class GenerateDataFromHtml:
    # jobPortalName = 'jobs-europe' #change for other job portal name. e.g euro-jobs
    # columnName = 'desccription2' # change for other job portal. e.g for euro-jobs the column name is 'description2'
    # fileName = 'jobs-europe-all'

    jobPortalName = 'euro-jobs'  # change for other job portal name. e.g euro-jobs
    columnName = 'html_blob'  # change for other job portal. e.g for euro-jobs the column name is 'description2'
    fileName = jobPortalName + '-data'
    googleTransCharLimit = 5000
    translator = google_translator()


    def _init_(self):

        '''impoting sample eurojobs data'''
        #self.generateDataAsChunk()
        #self.generateParsedChunk()
        self.generatePlainTextData()

    '''---------------------------------------START OF GENERIC METHODS---------------------------------------------'''
    '''Method for removing some unwanted html tags and strigs'''

    def removeUnwantedStr(self, str):
        unwantedStr = ['<strong>', '</strong>', '<b>', '</b>', '<h1>', '</h1>', '<h2>', '</h2>', '<h3>', '</h3>',
                       '<h4>', '</h4>', '<h5>', '</h5>', '<h6>', '</h6>', '&amp;']
        for i in unwantedStr:
            str = str.replace(i, '')
        return str

    '''method for cleaning string'''

    def clean_string(self, data):

        custom_string_punctuation = re.sub(r"[$-:.%,£_]", "",string.punctuation)  # removing some element from default string punctuation

        try:
            data = unicodedata.normalize("NFKD", data)
            data = data.strip()
            data = re.sub(r'[\n\r\t]', '', data)
            data = data.lower()
            data = re.sub("([^-_a-zA-Z0-9!@#%&=,'\";:~`č₽€£\$\^\(\)\+\[\]\.\{\}\|\?\<\>\\]+|[^\s]+)", '', data)
            data = re.sub(r'<[^>]+>', '', data)  # removing html tags
            data = re.sub(r'a€™|a€tm|a€¦|a€|a€“|a€¢|â€™|â€tm|â€¦|â€|â€“|â€¢|i€', '', data)
            data = re.sub(r'\^[a-zA-Z]\s+', '', data)  # Remove single characters from the start
            data = re.sub(r'\s+', ' ', data, flags=re.I)  # Substituting multiple spaces with single space
            data = re.sub(r'^b\s+', '', data)  # Removing prefixed 'b'
            data = data.replace('&amp;', 'and')
            data = data.replace('atm', '')  # for removing atm from all masteratms or membersatm string
            data = re.sub(r"([a-zA-Z])(\d+)", r"\g<1> \g<2>", data)  # a34301 => 34301
            data = re.sub(r"\b[a-zA-Z]\b", "", data)  # removing single caracter
            data = re.sub(r"(\d+)(k)", r"\g<1>000", data)
            data = re.sub('(?<=\d),(?=\d{3})', '', data)  # 6,576,570 => 6576570
            data = data.replace(' euro ', ' € ')
            data = data.replace(' eur ', ' € ')
            data = data.replace(' dollar ', ' $ ')
            data = data.replace(' usd ', ' $ ')

            data = re.sub(r"(\d+).(\d+) (€)|(\d+).(\d+)(€)|(\d+) (€)|(\d+)(€)", r"€ \g<1>", data)  # 2984 € => € 2984
            data = re.sub(r"(\d+).(\d+) (\$)|(\d+).(\d+)(\$)|(\d+) (\$)|(\d+)(\$)", r"$ \g<1>",data)  # 2984 $ => $ 2984
            data = re.sub(r"(\d+).(\d+) (£)|(\d+).(\d+)(£)|(\d+) (£)|(\d+)(£)", r"£ \g<1>", data)  # 2984 £ => £ 2984

            data = re.sub('(?<=\d{4})-(?=\d{4})', ' to ', data)  # 1000-2000=> 1000 to 20000 for NER product to cardinal
            data = re.sub(r"[" + custom_string_punctuation + "]", "",data)  # removing some string puntuations ['!"#;<=>?@[\\]^_`{|}~']
            
            # Lemmatization
            data = data.split()
            data = [WordNetLemmatizer().lemmatize(word) for word in data]
            data = ' '.join(data)
            return data

        except Exception as e:

            print('clean string failure !!!', data, e)
            return ''

    '''---------------------------------------END OF GENERIC METHODS---------------------------------------------'''

    '''-----------------------START OF METHODS FOR PREPARE CHUNKS FROM HTML DOCUMENTS----------------------------'''
    '''method for generating parsed text chunk from individual datapoint html document'''

    
    def detectLanguage(self, text):
        try:
            detectResult = self.translator.detect(text[-200:])
            return detectResult[0]
        except:
            return False
        
        
    def split_list(self,a_list):
        half = len(a_list)//2
        return a_list[:half], a_list[half:]


    
    def translateChunkedData(self,chunkedData,language):

        if language != 'en':
            charLen = len(str(chunkedData))
            if charLen < self.googleTransCharLimit:
                data = self.translator.translate(chunkedData)
                data = re.sub(r"([a-zA-Z])'([a-zA-Z])", r"\g<1>\g<2>", data)
                data = re.sub(r'([a-zA-Z])"([a-zA-Z])', r"\g<1>\g<2>", data)
                data = ast.literal_eval(data)
            else:
                split_1, split_2 = self.split_list(chunkedData)
                split_1 = self.translateChunkedData(split_1,language)
                split_2 = self.translateChunkedData(split_2,language)
                data = split_1 + split_2
        else:
            data = chunkedData

        return data
    
    def generateDataAsChunk(self):

        '''time tracking start'''
        timeStart = datetime.datetime.now()
        print('generateDataAsChunk start:', timeStart)

        #translator = google_translator()

        if self.jobPortalName == 'jobs-europe':
            df = pd.read_excel(open('../data/' + self.jobPortalName + '/' + self.fileName + '.xlsx', 'rb'),
                               sheet_name='data')
        else:  # for euro-jobs
            df = pd.read_csv('../data/' + self.jobPortalName + '/' + self.fileName + '.csv')
            
        #print('total: ', len(df))
        df = df[df[self.columnName].notnull()]
        df.reset_index(drop=True, inplace = True)
        df['id'] = df.index + 1
        #print('after removing nan total: ', len(df))
        #df = df.head(1000)
        '''functionality for extracting text data from html document of sample data'''
        parsedData = []
        for index, row in df.iterrows():
            try:
                dataPointId = row['id']

                totalHtml = row[self.columnName]
                
                # removing newline, tab from html document
                totalHtml = unicodedata.normalize("NFKD", totalHtml)
                totalHtml = totalHtml.strip()
                totalHtml = re.sub(r'[\n\r\t]', '', totalHtml)

                data = self.removeUnwantedStr(totalHtml)

                # functinality for structinng text from html document
                customParser = MyCustomHTMLParser()
                customParser.chunkedData = []
                customParser.origChunkedData = []
                customParser.feed(data)
                parsedData.append(
                    [dataPointId, customParser.chunkedData, customParser.origChunkedData, totalHtml])
                print('success !!!', dataPointId)
                print('-----------------------------------------------------------------------------------------------')
                # break

            except Exception as e:
                print('generateDataAsChunk failure !!!', dataPointId, e)

        df = pd.DataFrame(parsedData)
        df.columns = ['id', 'chunked_data', 'orig_chunked_data', 'total_html']
        df['chunk_count'] = df['chunked_data'].apply(lambda x: len(x))
        #df['language'] = df['chunked_data'].apply(lambda x: self.detectLanguage(str(x)))
        #df['translated_chunks'] = df.apply(lambda x: translateChunkedData(x.chunked_data,x.language),  axis=1)

        df.to_csv(r'../data/parsed-data/' + self.jobPortalName + '-parsed.csv', index=False)

        print('generateDataAsChunk time Required', datetime.datetime.now() - timeStart)  # time tracking end

        return df

    '''method for generating parsed chunks'''

    def generateParsedChunk(self):
        # TODO: parallelize google translate queries - progress bar

        '''time tracking start'''
        timeStart = datetime.datetime.now()
        print('generateParsedChunk start:', timeStart)

        df = pd.read_csv(r'../data/parsed-data/' + self.jobPortalName + '-parsed.csv')
        parsedChunks = []
        chunkId = 1
        for index, row in df.iterrows():
            dataPointId = row['id']
            #try:
            #data = row['translated_chunks']
            #data = data.replace('] [', ',')
            #data = re.sub(r"','|' ,'|', '|' , '", "SPLITMEHERE", data)
            #data = data.replace("[", '')
            #data = data.replace("']", '')
            #data = data.replace("'", '')
            #data = data.replace('"', '')
            #chunks = data.split('SPLITMEHERE')
            
            chunks = ast.literal_eval(row['chunked_data'])
            for chunk in chunks:
                if len(chunk.split()) > 1:
                    parsedChunks.append([dataPointId, chunkId, chunk])
                    chunkId += 1
            print('generateParsedChunk success !!! dataPointId: ', dataPointId)
            #except Exception as e:
            #    print('generateParsedChunk failure !!!', dataPointId, e)
            #    continue

        df = pd.DataFrame(parsedChunks, columns=['data_point_id', 'chunk_id', 'chunk'])
        df.to_csv(r'../data/parsed-data/' + self.jobPortalName + '-chunks.csv', index=False)

        print('generateParsedChunk time Required', datetime.datetime.now() - timeStart)  # time tracking end

        return df

    def translate(self, text):
        try:
            detectResult = self.translator.detect(text)
            if detectResult and 'en' not in detectResult:
                print(detectResult)
                print('non translated text:', text)
                text = translator.translate(text)
                print('translated text:', text)
        except e:
            print(e)
        #time.sleep(1)
        return text

    '''method for generatng plain text from whole html document'''
    
    def plainTextFromChunk(self,data):
        
        data = self.clean_string(data)
        data = data.replace('] [', ',')
        data = re.sub(r"','|' ,'|', '|' , '", '. ', data)
        data = data.replace("[", '')
        data = data.replace("']", '')
        data = data.replace("'", '')
        data = data.replace('"', '')
        return data
    
    def generatePlainTextData(self):

        '''time tracking start'''
        timeStart = datetime.datetime.now()
        print('generatePlainTextData start:', timeStart)
        df = pd.read_csv(r'../data/parsed-data/' + self.jobPortalName + '-parsed.csv')
        df['job_desc_plain'] = df['chunked_data'].apply(lambda x: self.plainTextFromChunk(x))
        #df['job_desc_plain'] = df['translated_chunks'].apply(lambda x: self.plainTextFromChunk(x))
        df.to_csv(r'../data/parsed-data/' + self.jobPortalName + '-refined-html.csv', index=False)
        print('generatePlainTextData time Required', datetime.datetime.now() - timeStart)  # time tracking end

        return df

    def generatePlainTextDataOld(self):

        '''time tracking start'''
        timeStart = datetime.datetime.now()
        print('generatePlainTextData start:', timeStart)

        #translator = google_translator()

        if self.jobPortalName == 'jobs-europe':
            df = pd.read_excel(open('../data/' + self.jobPortalName + '/' + self.fileName + '.xlsx', 'rb'),
                               sheet_name='data')
        else:  # for euro-jobs
            df = pd.read_csv('../data/' + self.jobPortalName + '/' + self.fileName + '.csv')

        df['job_desc_plain'] = df[self.columnName].apply(lambda x: self.clean_string(string(x)))
        #df['non_translated_job_desc_plain'] = df[self.columnName].apply(lambda x: self.clean_string(x))
        #df['job_desc_plain'] = df['non_translated_job_desc_plain'].apply(lambda x: self.translate(x))
        df.to_csv(r'../data/parsed-data/' + self.jobPortalName + '-refined-html.csv', index=False)

        print('generatePlainTextData time Required', datetime.datetime.now() - timeStart)  # time tracking end

        return df

    '''--------------------------END OF METHODS FOR PREPARE CHUNKS FROM HTML DOCUMENTS----------------------------'''

GenerateDataFromHtml()._init_()

