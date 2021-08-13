#!/usr/bin/env python
# coding: utf-8

# In[17]:


'''Importing libraries'''
import multiprocessing
import os
import sqlite3

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import datetime
import unicodedata
import re
import string
import ast
import en_core_web_lg

nlp = en_core_web_lg.load()
import pickle
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
from nltk.corpus import stopwords
from html.parser import HTMLParser
from nltk.stem import WordNetLemmatizer

stopwords = stopwords.words('english')
from tqdm import tqdm

import requests  # v = 2.24.0
import json

TEMP_PATH = "../data/temp_data/"


class customTranslate:
    headers = {
        'x-rapidapi-key': "beec17bc18mshd714ed855e68198p10e39djsn603089bdfa69",
        'x-rapidapi-host': "systran-systran-platform-for-language-processing-v1.p.rapidapi.com"
    }
    apiUrl = "https://systran-systran-platform-for-language-processing-v1.p.rapidapi.com"

    def detect(self, text):
        if re.search('[a-zA-Z]', text) and len(text) > 1:
            m = round(len(text) / 2)  # middle caracter number
            text = text[m - 50: m + 50]  # take only 100 middle characters
            response = requests.request("GET", headers=self.headers,
                                        params={"input": text},
                                        url=self.apiUrl + "/nlp/lid/detectLanguage/document"
                                        )
            # print(response.text)
            if response.status_code == 200:
                return json.loads(response.text)['detectedLanguages'][0]['lang']

        return False

    def translate(self, text, lang='auto'):
        if str(lang) not in ['en', 'False'] and re.search('[a-zA-Z]', text) and len(text) > 1:
            response = requests.request("GET", headers=self.headers,
                                        params={"source": lang, "target": "en", "input": text},
                                        url=self.apiUrl + "/translation/text/translate"
                                        )
            # print(response.text)
            if response.status_code == 200:
                # print(response.text)
                return json.loads(response.text)['outputs'][0]['output']

        return text


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
    transCharLimit = 5000

    translator = customTranslate()

    def _init_(self, display):
        # pandarallel.initialize(nb_workers=512)
        # self._counter = 0

        display.set_state(1)

        '''impoting sample eurojobs data'''
        self.generateDataAsChunk(display)
        self.generateParsedChunk(display)

        display.set_state(0)

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

        custom_string_punctuation = re.sub(r"[$-:.%,£_]", "",
                                           string.punctuation)  # removing some element from default string punctuation

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
            data = re.sub(r"(\d+).(\d+) (\$)|(\d+).(\d+)(\$)|(\d+) (\$)|(\d+)(\$)", r"$ \g<1>",
                          data)  # 2984 $ => $ 2984
            data = re.sub(r"(\d+).(\d+) (£)|(\d+).(\d+)(£)|(\d+) (£)|(\d+)(£)", r"£ \g<1>", data)  # 2984 £ => £ 2984

            data = re.sub('(?<=\d{4})-(?=\d{4})', ' to ', data)  # 1000-2000=> 1000 to 20000 for NER product to cardinal
            data = re.sub(r"[" + custom_string_punctuation + "]", "",
                          data)  # removing some string puntuations ['!"#;<=>?@[\\]^_`{|}~']

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

    def split_list(self, a_list):
        half = len(a_list) // 2
        return a_list[:half], a_list[half:]

    def translateChunkedData(self, data, language):
        if str(language) not in ['en', 'False']:
            data = '||'.join(data)
            charLimit = self.transCharLimit
            charLen = len(data)
            if charLen > charLimit:
                n = int(np.ceil(charLen / charLimit))
                translatedData = ''
                for i in range(n):
                    dt = data[i * charLimit:(i + 1) * charLimit]
                    translatedData += self.translator.translate(dt, language)
            else:
                translatedData = self.translator.translate(data, language)

            data = translatedData.split('||')

        return data

    def _detect(self, text):
        id = text[self._id_index]
        if id in self._done:
            return None
        text = text[self._chunked_index]
        try:
            return id, self.translator.detect(' '.join(text))
        except:
            return None

    def _translate(self, text):
        id = text[self._id_index]
        if id in self._done:
            return None
        try:
            language = text[self._language_index]
            text = text[self._chunked_index]
            return id, self.translateChunkedData(text, language)
        except:
            return None

    def generateDataAsChunk(self, display):
        '''time tracking start'''
        timeStart = datetime.datetime.now()
        # print('generateDataAsChunk start:', timeStart)

        #if self.jobPortalName == 'jobs-europe':
        #    df = pd.read_excel(open('../data/' + self.jobPortalName + '/' + self.fileName + '.xlsx', 'rb'),
        #                       sheet_name='data')
        #else:  # for euro-jobs
        #    df = pd.read_csv('../data/' + self.jobPortalName + '/' + self.fileName + '.csv')

        con = sqlite3.connect('../scraping/scrapedata.db')

        df = pd.read_sql_query("SELECT * FROM myscrapedata", con)

        df = df[df[self.columnName].notnull()]
        df.reset_index(drop=True, inplace=True)
        df['id'] = df.index + 1

        # df = df.head(100)

        '''functionality for extracting text data from html document of sample data'''
        parsedData = []
        self._id_index = df.columns.get_loc('id')
        self._url_index = df.columns.get_loc('url')
        self._column_index = df.columns.get_loc(self.columnName)

        display.set_state(2)

        with multiprocessing.Pool(processes=os.cpu_count()) as p:
            with tqdm(total=len(df), unit=' job postings') as p_bar:
                for i, res in enumerate(p.imap(self.parallel_chunking, df.values.tolist())):
                    p_bar.update()
                    parsedData.append(res)

        df = pd.DataFrame(parsedData)
        df.columns = ['id', 'url', 'chunked_data', 'orig_chunked_data', 'total_html']

        df['chunk_count'] = df['chunked_data'].apply(lambda x: len(x))
        tqdm.pandas(unit=" chunks")

        display.set_state(3)
        # print("Detecting Language")
        if os.path.isfile(TEMP_PATH + "language.pickle"):
            with open(TEMP_PATH + "language.pickle", "rb") as file:
                detected = pickle.load(file)
                self._done = set(detected.keys())
        else:
            detected = {}
            self._done = set([])
        self._chunked_index = df.columns.get_loc('chunked_data')
        with multiprocessing.Pool(processes=os.cpu_count()) as p:
            with tqdm(total=len(df), unit=' job postings') as p_bar:
                for i, res in enumerate(p.imap_unordered(self._detect, df.values.tolist())):
                    p_bar.update()
                    if res is not None:
                        detected[res[0]] = res[1]
                        if i % 200 == 0:
                            with open(TEMP_PATH + "language.pickle", "wb") as file:
                                pickle.dump(detected, file, protocol=pickle.HIGHEST_PROTOCOL)
        df['language'] = df['id'].map(detected)
        del (detected)

        self._language_index = df.columns.get_loc('language')
        # print("Translating")
        display.set_state(4)
        if os.path.isfile(TEMP_PATH + "translation.pickle"):
            with open(TEMP_PATH + "translation.pickle", "rb") as file:
                translated = pickle.load(file)
                self._done = set(translated.keys())
        else:
            translated = {}
            self._done = set([])
        with multiprocessing.Pool(processes=os.cpu_count()) as p:
            with tqdm(total=len(df), unit=' job postings') as p_bar:
                for i, res in enumerate(p.imap_unordered(self._translate, df.values.tolist())):
                    p_bar.update()
                    if res is not None:
                        translated[res[0]] = res[1]
                        if i % 1 == 0:
                            with open(TEMP_PATH + "translation.pickle", "wb") as file:
                                pickle.dump(translated, file, protocol=pickle.HIGHEST_PROTOCOL)
        # df['translated_chunks'] = df['id'].map(translated)
        df['translated_chunks'] = df['id'].apply(lambda x: translated.get(x, []))
        del (translated)
        # df.insert(len(df.columns), 'translated_chunks', translated)

        # df['translated_chunks'] = df.parallel_apply(self._translate, axis=1)
        # df['job_desc_plain'] = df['chunked_data'].apply(lambda x: '. '.join(x))
        df['job_desc_plain'] = df['translated_chunks'].apply(lambda x: '. '.join(x))

        df.to_csv(r'../data/parsed-data/' + self.jobPortalName + '-parsed.csv', index=False)

        # print('generateDataAsChunk time Required', datetime.datetime.now() - timeStart)  # time tracking end

        return df

    def parallel_chunking(self, row):

        dataPointId = row[self._id_index]
        dataUrl = row[self._url_index]
        totalHtml = row[self._column_index]

        # removing newline, tab from html document
        totalHtml = unicodedata.normalize("NFKD", totalHtml)
        totalHtml = totalHtml.strip()
        totalHtml = re.sub(r'[\n\r\t]', '', totalHtml)

        data = self.removeUnwantedStr(totalHtml)

        # functinality for structinng text from html document
        customParser = MyCustomHTMLParser()  # TODO check if custom parser needs to be initialised every time
        customParser.chunkedData = []
        customParser.origChunkedData = []
        customParser.feed(data)
        return [dataPointId, dataUrl, customParser.chunkedData, customParser.origChunkedData, totalHtml]

    '''method for generating parsed chunks'''

    def generateParsedChunk(self, display):
        display.set_state(5)

        '''time tracking start'''
        timeStart = datetime.datetime.now()
        # print('generateParsedChunk start:', timeStart)

        df = pd.read_csv(r'../data/parsed-data/' + self.jobPortalName + '-parsed.csv')
        parsedChunks = []
        chunkId = 1

        with tqdm(total=len(df), unit=' job postings') as p_bar:
            for index, row in df.iterrows():
                dataPointId = row['id']
                # chunks = ast.literal_eval(row['chunked_data'])
                chunks = ast.literal_eval(row['translated_chunks'])
                for chunk in chunks:
                    if len(chunk.split()) > 1:
                        parsedChunks.append([dataPointId, chunkId, chunk])
                        chunkId += 1
                p_bar.update()

        df = pd.DataFrame(parsedChunks, columns=['data_point_id', 'chunk_id', 'chunk'])
        df.to_csv(r'../data/parsed-data/' + self.jobPortalName + '-chunks.csv', index=False)

        # print('generateParsedChunk time Required', datetime.datetime.now() - timeStart)  # time tracking end

        return df

    '''--------------------------END OF METHODS FOR PREPARE CHUNKS FROM HTML DOCUMENTS----------------------------'''


if __name__ == '__main__':

    GenerateDataFromHtml()._init_()

    # In[18]:

    df = pd.read_csv('../data/parsed-data/euro-jobs-parsed.csv')
    # df = pd.read_csv('../data/parsed-data/euro-jobs-chunks.csv')
    # df = pd.read_csv('../data/euro-jobs/euro-jobs-data.csv')
    df

    # In[19]:

    for i, j in df.iterrows():
        print(j['job_desc_plain'])

    # In[ ]:
