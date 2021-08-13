#!/usr/bin/env python
# coding: utf-8

# In[76]:


'''importing libraries'''

import en_core_web_lg
import pandas as pd

nlp = en_core_web_lg.load()
# spacy.require_gpu()

import string
import geograpy
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
from nltk.corpus import stopwords

stopwords = stopwords.words('english')
import re
from nltk.stem import WordNetLemmatizer
import unicodedata
from sklearn.feature_extraction.text import CountVectorizer
from word2number import w2n
from tqdm import tqdm


class ExtractInfo:
    jobPortal = 'euro-jobs'
    refinedHtmlFile = jobPortal + '-refined-html'
    nonLabeledChunkFile = jobPortal + '-chunks'
    labeledChunkFile = jobPortal + '-labeled-chunks'

    jkb_countries = []

    '''----------------------START OF VARIABLE DECLARATION FOR EMPLOYMENT TYPE INFORMATION EXTRACTION------------------'''
    full_time = ['full time', 'fulltime', 'permanent job', 'permanent position', 'fixed contract', 'fixed job',
                 'fixed term', 'fixed position', 'regular position', 'regular job']  # 11
    part_time = ['part time', 'parttime', 'oddjob', 'odd job', 'casual job', 'casual position', 'seasonal job',
                 'minijob', 'mini job', 'irregular job', 'temporary position', 'temporary job']  # 12
    intership = ['internship', 'internee', 'traineeship', 'trainee', 'apprentices', 'apprentice', 'apprenticeship']  # 7
    freelancing = ['freelance', 'freelancer', 'freelancing']  # 3
    zero_hour = ['zero hour']  # 1
    employementTypeCommonKeywords = full_time + part_time + intership + freelancing + zero_hour
    '''----------------------END OF VARIABLE DECLARATION FOR EMPLOYMENT TYPE INFORMATION EXTRACTION--------------------'''

    '''---------------------START OF VARIABLE DECLARATION FOR SALARY AND CURRENCY INFORMATION EXTRACTION---------------'''
    currencyList = []
    salaryCombinations = [
        ["hour", "income"], ["week", "income"], ["month", "income"], ["year", "income"], ["annual", "income"],
        ["hour", "salary"], ["week", "salary"], ["month", "salary"], ["year", "salary"], ["annual", "salary"],
        ["hour", "payment"], ["week", "payment"], ["month", "payment"], ["year", "payment"], ["annual", "payment"]
    ]
    salaryCommonKeywords = ['remuneration', 'salary', 'compensation', 'earnings', 'emolument', 'gross', 'incentive',
                            'paycheck', 'remunerated', 'reward', 'stipend', 'salaries', 'wages']
    durationList = ["month", "year", "annual""hour", "week"]
    monthNames = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
                  'november', 'december', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov',
                  'dec']
    '''---------------------END OF VARIABLE DECLARATION FOR SALARY AND CURRENCY INFORMATION EXTRACTION-----------------'''

    '''---------------------START OF VARIABLE DECLARATION FOR WORK HOUR INFORMATION EXTRACTION-------------------------'''
    workHoursCombinations = [
        ["day", "hour"], ["days", "hour"], ["day", "hours"], ["days", "hours"], ["week", "hour"], ["weeks", "hour"],
        ["week", "hours"], ["weeks", "hours"], ["month", "hour"], ["months", "hour"], ["month", "hours"],
        ["months", "hours"],
        ["full time", "hours"], ["full time", "hour"], ["full time", "hour"]
    ]
    singleWordListForWorkHours = ["day", "hour", "days", "hours", "week", "weeks", "month", "months", "year", "years",
                                  "working hour", "weekend", "daily", "hourly", "weekly", "monthly", "yearly"]
    workHoursCommonKeywords = [
        'business hours', 'company time', 'duty times', 'employment period', 'five day week', 'hours of work',
        'hours work',
        'hours worked', 'hours of employment', 'labor hours', 'length of time worked', 'man hours', 'office hours',
        'office hour', 'overtime', 'regular time', 'regular timetable', 'regular hours',
        'staff time', 'working hours', 'working hour', 'working hour can be discussed', 'working time', 'work schedule',
        'work time', 'workday', 'working day', 'working periods', 'work days', 'workdays', 'flexible with working hour'
    ]
    textTypeWorkHours = [
        'working hour can be discussed', 'flexible with working hour', 'flexible working hour',
        'flexible office hour', 'flexible operation hour', 'flexible business hour', 'regular working hour',
        'regular office hour', 'regular operation hour', 'regular business hour', 'normal working hour',
        'normal office hour', 'normal operation hour', 'normal business hour',
        'standard working hour', 'standard office hour', 'standard operation hour',
        'standard business hour', 'overnight', 'basis hour', 'morning', 'noon', 'night', 'evening'
    ]
    defaultNormalHoursStings = ['regular working hour', 'regular office hour', 'regular operation hour',
                                'regular business hour',
                                'normal working hour', 'normal office hour', 'normal operation hour',
                                'normal business hour',
                                'standard working hour', 'standard office hour', 'standard operation hour',
                                'standard business hour'
                                ]
    '''---------------------END OF VARIABLE DECLARATION FOR WORK HOUR INFORMATION EXTRACTION---------------------------'''

    def _init_(self, PARAMETERS, display):

        #display.set_state(11)

        # csv file list of degree list
        #self.degree_list_csv = PARAMETERS['degree_list_csv']

        '''importing country list'''
        # europian_countries = pd.read_csv("../data/europian_countries.csv")
        #jkb_countries_df = pd.read_csv("../data/essential/jkb_countries.csv", sep=';')
        #self.jkb_countries = list(jkb_countries_df['country'].values)
        #self.jkb_countries.extend(('United Kingdom', 'England', 'Etaly', 'Iceland', 'Liechtehenstein',
        #                           'Switzerland'))  # adding some countries out of jkb country list
        # print(self.jkb_countries)
        '''importing currency list'''
        #currencyListDf = pd.read_csv("../data/essential/currecy-list.csv")
        #self.currencyList = list(map(str.lower, list(currencyListDf['currency'].values)))
        #self.currencyList.extend(('$', '€', '£'))
        # print(self.currencyList)
        '''importing chunk dataset'''
        #df = self.generateDocData(self.labeledChunkFile, 'labeled-data', display)  # for labeled data
        # df = self.generateDocData(self.nonLabeledChunkFile,'parsed-data')# for non labeled data

        # sys.exit(0)
        '''un comment those categories which need to extract'''

        # print("Extracting location")

        #display.set_state(14)

        #self.extractInfoDependentOnNER(df, 'GPE', 'job_location')
        # self.extractInfoDependentOnNER(df[df['category'] == 'start_date'],'DATE','start_date')
        # self.extractInfoDependentOnNER(df[df['category'] == 'deadline_date'],'DATE','deadline_date')
        # self.extractInfoDependentOnNER(df[df['category'] == 'hiring_organization'],'ORG','hiring_organization')

        #display.set_state(15)
        # print("Extracting educational requirements")
        #self.extractingEducationRequirements(df[df['category'] == 'education_requirements'])

        #display.set_state(16)
        # print("Extracting employment type")
        #self.employmentTypeExtraction(df)

        #display.set_state(17)
        # print("Extracting base salary")
        self.extractingBaseSalary(df)

        self.extractWorkHours(display)

        display.set_state(0)

    '''---------------------------------------START OF GENERIC METHODS---------------------------------------------'''
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
            data = re.sub(r'\^[a-zA-Z]\s+', '', data)  # Remove single characters from the start
            data = re.sub(r'\s+', ' ', data, flags=re.I)  # Substituting multiple spaces with single space
            data = re.sub(r'^b\s+', '', data)  # Removing prefixed 'b'
            data = data.replace('&amp;', 'and')
            data = data.replace('atm', '')  # for removing atm from all masteratms or membersatm string
            data = re.sub(r"([a-zA-Z])(\d+)", r"\g<1> \g<2>", data)  # a34301 => 34301
            data = re.sub(r"\b[a-zA-Z]\b", "", data)  # removing single caracter
            # data = re.sub(r".00", "", data)  # 300.00 => 300
            data = re.sub(r"(\d+)(k)", r"\g<1>000", data)
            data = re.sub('(?<=\d),(?=\d{3})', '', data)  # 6,576,570 => 6576570
            # data = re.sub('(?<=\d) (?=\d)', ' ', data)
            # data = re.sub('(?<=[a-zA-Z]),(?=[a-zA-Z\s])', ' ', data)
            # data = re.sub('(?<=\d) (?=\d{3})', '', data)
            # cleaned_chunk = re.sub('(?<=\d) (?=\d)', '.', cleaned_chunk)
            # data = re.sub('(?<=\d) (?=\d)', '-', data)
            # data = re.sub(r"(\d+)-(00)", r"\g<1>", data) # e.g 34567-00 => 34567
            data = data.replace('euro ', '€')
            data = data.replace('eur ', '€')
            data = data.replace('dollar ', '$')
            data = data.replace('usd ', '$')
            data = re.sub(r"(\d+) (€)", r"€ \g<1>", data)  # 2984 € => € 2984
            data = re.sub(r"(\d+)(€)", r"€ \g<1>", data)  # 2984€ => € 2984
            data = re.sub(r"(\d+) (\$)", r"$ \g<1>", data)  # 2984 $ => $ 2984
            data = re.sub(r"(\d+)(\$)", r"$ \g<1>", data)  # 2984$ => $ 2984
            data = re.sub(r"(\d+) (£)", r"£ \g<1>", data)  # 2984 € => € 2984
            data = re.sub(r"(\d+)(£)", r"£ \g<1>", data)  # 2984€ => € 2984
            data = re.sub('(?<=\d{4})-(?=\d{4})', ' to ', data)  # 1000-2000=> 1000 to 20000 for NER product to cardinal
            data = re.sub(r"[" + custom_string_punctuation + "]", "",
                          data)  # removing some string puntuations ['!"#;<=>?@[\\]^_`{|}~']
            # data = word_tokenize(data)
            # print(data)
            # data = ' '.join([word for word in data if word not in custom_string_punctuation])
            # data = ' '.join([word for word in data.split() if word not in stopwords])
            # Lemmatization
            data = data.split()
            data = [WordNetLemmatizer().lemmatize(word) for word in data]
            data = ' '.join(data)
            # while re.match(r'^[a-zA-Z0-9]\s+', document):
            # document = re.sub(r'^[a-zA-Z0-9]\s+', '', document)
            return data
        except:
            return ''

    '''method for matching multiple element of a list with multiple words in a string'''

    def getMatchedElementOfListWithStringMultiple(self, _list, _str):
        resp = []
        for i in _list:
            if i in _str and i not in resp:
                resp.append(i)
        if resp:
            return resp
        else:
            return False

    '''metod for word to number conversion'''

    def wordToNumberStr(self, _str):
        ret = []
        for word in _str.split():
            try:
                ret += [str(w2n.word_to_num(word))]
            except ValueError:
                ret += [word]
        return ' '.join(ret)

    '''method for prepare conditions dependent on combinations'''

    def getOrConditionUsingCombinations(self, combinations, chunk):
        condition = False
        for combination in combinations:
            condition = condition or all(x in chunk for x in combination)
        return condition

    '''method for getting matched elements from a string'''

    def getMatchedElementOfListWithString(self, _list, _str, exact_match=False):
        if exact_match:
            _str = _str.split()
        for i in _list:
            if i in _str:
                return i
        return False

    '''function to return keys for any value'''

    def get_keys(self, my_dict, val):
        keys = []
        for key, value in my_dict.items():
            if val == value:
                keys.append(key)
        return keys

    '''method for getting values with minimum distance'''

    def minDistance(self, lst):
        lst = sorted(lst)
        index = -1
        distance = max(lst) - min(lst)
        res = []
        for i in range(len(lst) - 1):
            if lst[i + 1] - lst[i] < distance:
                distance = lst[i + 1] - lst[i]
                index = i
        for i in range(len(lst) - 1):
            if lst[i + 1] - lst[i] == distance:
                # print (lst[i],lst[i+1])
                res.append(lst[i])
                res.append(lst[i + 1])
        return res

    '''method for finding n number of surronding words of a keyword'''

    def findSurroundingWords(self, _str, key, n):
        sub = '\w*\W*' * n + re.escape(key) + '\W*\w*' * n
        nearbyWords = []
        for i in re.findall(sub, _str, re.I):
            nearbyWords.append(i)
        return ' '.join(nearbyWords)

    '''-----------------------------------------END OF GENERIC METHODS---------------------------------------------'''

    '''-----------------START OF GENERIC METHODS FOR EXTRACTING DIFFERENT INFORMATION------------------------------'''
    '''method for generating nlp doc for  allLabeledData'''

    def generateDocData(self, dataFile, directory, display):
        df = pd.read_csv("../data/" + directory + "/" + dataFile + ".csv")
        # df = df.head(50000)
        # df = df[df['data_point_id']==200]
        # print("Cleaning chunks")
        display.set_state(12)
        tqdm.pandas(unit=" chunks")
        df['cleaned_chunk'] = df['chunk'].progress_apply(lambda x: self.clean_string(x))
        self._id_index = df.columns.get_loc('chunk_id')
        self._col_index = df.columns.get_loc('cleaned_chunk')
        nlp_d = {}

        # print("Generating features")
        display.set_state(13)

        with tqdm(total=len(df), unit=' job postings') as p_bar:
            for text in df.values.tolist():
                p_bar.update()
                res = self._parallel_nlp(text)
                nlp_d[res[0]] = res[1]
        df['doc'] = df['chunk_id'].map(nlp_d)

        # df['doc'] = df['cleaned_chunk'].apply(lambda x: nlp(x))
        df = df.reset_index(drop=True)
        # df.to_csv(r'../data/extracted-data/nlp-doc/' + dataFile + '-nlp-doc.csv', index=False)
        return df

    def _parallel_nlp(self, text):
        id = text[self._id_index]
        to_nlp = text[self._col_index]
        return id, nlp(to_nlp)

    '''method for extracting information using spacy named entity features'''

    def extractInfoDependentOnNER(self, df, NER_type, category):
        # print('Total instance number of '+category+' category in allLabeledDf data',len(df[df['category'] == category]))
        data = []
        with tqdm(total=len(df), unit=' chunks') as p_bar:
            for index, row in df.iterrows():
                doc = row['doc']
                chunk = row['chunk']
                cleaned_chunk = row['cleaned_chunk']

                info = []
                for X in doc.ents:
                    if X.text not in info and X.label_ == NER_type:

                        if NER_type == 'GPE':  # for location type info
                            # if self.checkLocationConditions(X.text):
                            placeName = self.placeExtractingUsingGeography(X.text)
                            # print(placeName)
                            for i in placeName:
                                # if i not in info:
                                info.append(i)
                        elif NER_type == 'ORG':  # for hiring_organization type info
                            if self.checkOrganisationConditions(X.text):
                                info.append(X.text)
                        elif NER_type == 'DATE':  # for Date type info
                            if self.checkDateConditions(X.text):
                                info.append(X.text)
                        else:
                            info.append(X.text)

                if info:
                    data.append([row['data_point_id'], row['chunk_id'], chunk, cleaned_chunk, info, row['category']])

                p_bar.update()

        resDf = pd.DataFrame(data,
                             columns=['data_point_id', 'chunk_id', 'chunk', 'cleaned_chunk', category, 'category'])
        resDf.to_csv(r'../data/extracted-data/extracted_' + category + '.csv', index=False)

        return resDf

    '''-------------------END OF GENERIC METHODS FOR EXTRACTING DIFFERENT INFORMATION------------------------------'''

    '''-----------------START OF METHODS FOR EXTRACTING SALARY AND CURRENCY INFORMATION----------------------------'''
    '''method for extarcting currency'''

    def getCurrency(self, str_):
        # currency = self.getMatchedElementOfListWithString(self.currencyList, str_, True)
        currency = list(set(self.currencyList).intersection(str_.split()))
        currency_temp = self.getMatchedElementOfListWithString(['€', '$', '£'], str_, False)

        cur = False
        if currency:
            cur = currency[0]
        elif currency_temp:
            cur = currency_temp

        if cur:
            cur = re.sub(r'€|€ | €| € ', 'euro', cur)
            cur = re.sub(r'\$|\$ | \$| \$ ', 'usd', cur)
            cur = re.sub(r'£|£ | £| £ ', 'gbp', cur)

        return cur

    '''method for filtering currency and salary'''

    def filterCurrencyAndSalary(self, inf_c):
        salary = []
        currency = []
        salary_with_euro = self.get_keys(inf_c, 'euro')
        salary_with_euro_sign = self.get_keys(inf_c, '€')
        '''if any salary is associated with euro then that one is given priority'''
        if salary_with_euro and 'euro' not in currency:
            currency.append('euro')
            salary = salary_with_euro
        elif salary_with_euro_sign and 'euro' not in currency:
            currency.append('euro')
            salary = salary_with_euro_sign
        else:  # for other currency association all currency is returned
            currency = list(inf_c.values())
            salary = list(inf_c.keys())

        # print('filterCurrencyAndSalary', list(set(currency)), list(set(salary)))

        return list(set(currency)), list(set(salary))

    '''method for base salary extraction'''

    def extractingBaseSalary(self, df):
        # print('Total instance number of base_salary category in allLabeledDf data',len(df[df['category'] == 'base_salary']))
        # print("Extracting salary information")
        salaryCommonKeywords = list(pd.read_csv(r'../data/essential/salary_related_keywords.csv')['key_word'])
        data = []
        with tqdm(total=len(df), unit=' job postings') as p_bar:
            for i, row in df.iterrows():
                p_bar.update()
                chunk = row['chunk']
                cleaned_chunk = row['cleaned_chunk']
                doc = row['doc']

                '''we find for some combination string and some common string which determines that the chunk describes about salary'''
                if self.getOrConditionUsingCombinations(self.salaryCombinations, cleaned_chunk) or list(
                        set(salaryCommonKeywords).intersection(cleaned_chunk.split())) or self.getCurrency(
                        cleaned_chunk):
                    info = []
                    currencyInfo = []
                    salaryInfoWithCurrency = {}

                    '''if no number values found we discard those chunks'''
                    salaryInNumber = re.findall(r"([0-9.]*[0-9]+)", cleaned_chunk)
                    try:
                        salaryInNumber = [float(i) for i in salaryInNumber]
                    except:
                        continue

                    '''if the number value is very less we are discarding that. we are not allowing the value less than 450'''
                    salaryInNumber = [x for x in salaryInNumber if x >= 450]
                    # namedEntity = []
                    # print(salaryInNumber)
                    surrounding_words = []
                    for X in doc.ents:

                        '''we are allowing only three named entity types of spacy'''
                        if X.label_ in ['MONEY', 'CARDINAL', 'DATE'] and X.text not in info:
                            salaryTextToNum = re.findall(r"([0-9.]*[0-9]+)", X.text)
                            salaryTextToNum = [float(i) for i in salaryTextToNum]

                            for i in salaryTextToNum:

                                '''if extracted number from named entity feature does not exist in the numbers collected from the chunk then we are discarding that'''
                                if i in salaryInNumber and i not in info:
                                    # try:
                                    '''for date type we are trying to discard the year type strings'''
                                    # if X.label_ == 'DATE' and  1950 <= i <= 3000:
                                    if X.label_ == 'DATE':
                                        surroundingWords_2 = self.findSurroundingWords(cleaned_chunk, str(int(i)), 2)
                                        monthName = self.getMatchedElementOfListWithString(self.monthNames,
                                                                                           surroundingWords_2, True)
                                        '''if any month name found within the range of 2 words we are considering the number as a year value and discarding it'''
                                        if monthName:
                                            # print('year found: ',i,'cleaned_chunk: ',cleaned_chunk)
                                            continue

                                    # namedEntity.append([X.label_,X.text])
                                    # currency = parse_price(cleaned_chunk.upper()).currency

                                    '''we are finding the currency name among the nearby words of a number amount within a range of 5'''
                                    surroundingWords = self.findSurroundingWords(cleaned_chunk, X.text, 5)
                                    surrounding_words.append(surroundingWords)
                                    currency = self.getCurrency(X.text + surroundingWords)
                                    if currency and currency not in currencyInfo:
                                        currencyInfo.append(currency)

                                    if currency and currency not in salaryInfoWithCurrency:
                                        salaryInfoWithCurrency[i] = currency

                                    # annualSalary = self.getMatchedElementOfListWithString(['year','annual'],surroundingWords)
                                    # if annualSalary:
                                    # print('surroundingWords:',surroundingWords)

                                    info.append(i)

                                    # except:continue

                    if info:
                        # print('chunk: ', chunk)
                        # print('cleaned_chunk: ', cleaned_chunk)
                        # print('namedEntity: ', namedEntity)
                        # print('extarcted info: ', info)

                        '''If currency found in association with any number then the currency and salary is extracted separately'''
                        if len(info) > 1 and salaryInfoWithCurrency:
                            currencyInfo, info = self.filterCurrencyAndSalary(salaryInfoWithCurrency)

                        '''if more than 2 salary values are found then the closests values are kepet. and it is considered tant they are interval values'''
                        if len(info) > 2:  # choose closest two
                            info = self.minDistance(info)

                        currency_associated_salary_info = []
                        if currencyInfo:
                            currency_associated_salary_info = info

                        surrounding_words = ','.join(surrounding_words)
                        # if currencyInfo:
                        # print('final currency info', currencyInfo)

                        # print('------------------------------------')

                        data.append([row['data_point_id'], row['chunk_id'], chunk, cleaned_chunk, info, currencyInfo,
                                     currency_associated_salary_info, surrounding_words, row['category']])

        baseSalaryDf = pd.DataFrame(data, columns=['data_point_id', 'chunk_id', 'chunk', 'cleaned_chunk', 'base_salary',
                                                   'currency_info', 'currency_associated_salary_info',
                                                   'surrounding_words', 'category'])
        # print(baseSalaryDf)
        baseSalaryDf.to_csv(r'../data/extracted-data/extracted_base_salary.csv', index=False)

        return baseSalaryDf

    '''-------------------END OF METHODS FOR EXTRACTING SALARY AND CURRENCY INFORMATION----------------------------'''

    '''------------------------START OF METHODS FOR EXTRACTING JOB LOCATION INFORMATION----------------------------'''
    '''method for checking location conditions'''

    def checkLocationConditions(self, location_str):
        if len(location_str) < 4 and location_str not in ['uk',
                                                          'us']:  # we will not aalow a location name less than 4 caracter except uk, us
            return False
        return True

    '''method for extarcting location information using geography'''

    def placeExtractingUsingGeography(self, _str):
        _str = _str.title()  # geography detect the place name well which start with capital lettre that is why the string converted to title case

        if _str is None and _str == "":
            return []

        places = geograpy.get_place_context(text=_str)

        if places.countries:
            # print('extracted(not cleaned): ', places.countries)
            selected_countries = [x for x in places.countries if
                                  x in self.jkb_countries]  # keep only europian countries, discard others
            if selected_countries:
                # print('cleaned: ', selected_countries)
                # print('____________________________')
                return selected_countries
        # if places.countries or places.countries:
        # return places.countries+places.cities

        return []

    '''------------------------END OF METHODS FOR EXTRACTING JOB LOCATION INFORMATION------------------------------'''

    '''-----------------START OF METHODS FOR EXTRACTING EMPLOYMENT TYPE INFORMATION--------------------------------'''
    '''groupping employement type'''

    def getGroupNameOfEmployementType(self, employementType):
        resp = []
        for i in employementType:
            if i in self.full_time and 'full-time' not in resp:
                resp.append('full-time')
            elif i in self.part_time and 'part-time' not in resp:
                resp.append('part-time')
            elif i in self.intership and 'internship' not in resp:
                resp.append('internship')
            elif i in self.freelancing and 'freelancing' not in resp:
                resp.append('freelancing')
            elif i in self.zero_hour and 'zero-hour' not in resp:
                resp.append('zero-hour')
        return resp

    '''method for employement_type extraction'''

    def employmentTypeExtraction(self, df):
        data = []
        for i, row in df.iterrows():
            doc = row['doc']
            chunk = row['chunk']
            cleaned_chunk = row['cleaned_chunk']
            employementType = self.getMatchedElementOfListWithStringMultiple(self.employementTypeCommonKeywords,
                                                                             cleaned_chunk)

            if employementType:
                employementType = self.getGroupNameOfEmployementType(employementType)
                data.append(
                    [row['data_point_id'], row['chunk_id'], chunk, cleaned_chunk, employementType, row['category']])

        employmentTypeDf = pd.DataFrame(data, columns=['data_point_id', 'chunk_id', 'chunk', 'cleaned_chunk',
                                                       'employment_type', 'category'])
        employmentTypeDf.to_csv(r'../data/extracted-data/extracted_employment_type.csv', index=False)
        employmentTypeDf
        return employmentTypeDf

    '''-------------------END OF METHODS FOR EXTRACTING EMPLOYMENT TYPE INFORMATION--------------------------------'''

    '''-----------------START OF METHODS FOR EXTRACTING HIRING ORGANIZATION INFORMATION----------------------------'''
    '''method for checking organization name conditions'''

    def checkOrganisationConditions(self, org_str):
        avoidStr = ['http', 'https', 'www']
        if any(x in org_str for x in avoidStr):
            return False

        if len(org_str) < 3 and org_str not in ['eu',
                                                'un']:  # we will not allow a organisation name less than 4 caracter except eu, un
            return False

        return True

    '''-----------------END OF METHODS FOR EXTRACTING HIRING ORGANIZATION INFORMATION------------------------------'''

    '''-----------------START OF METHODS FOR EXTRACTING START AND DEADLINE DATE INFORMATION------------------------'''
    '''method for checking date conditions'''

    def checkDateConditions(self, date_str):
        if date_str in ['day', 'days', 'date', 'month', 'months', 'year', 'years']:
            return False
        return True

    '''-----------------END OF METHODS FOR EXTRACTING START AND DEADLINE DATE INFORMATION--------------------------'''

    '''-------------------START OF METHODS FOR EXTRACTING EDUCATIONAL REQUIREMENTS INFORMATION---------------------'''
    '''method for loading study program names'''

    def getStudyProgram(self):

        df = pd.DataFrame(columns=['degree'])
        for i in self.degree_list_csv:
            individual_degree_df = pd.read_csv(self.degree_list_csv[i])
            df = df.append(individual_degree_df, ignore_index=True)

        df['cleaned_study_name'] = df['degree'].apply(lambda x: self.clean_string(x))
        df = pd.DataFrame(df.cleaned_study_name.unique(), columns=['cleaned_study_name'])
        df = df.reset_index(drop=True)
        return df

    '''method for processing top 3 educational degree names'''

    def getTopEduReq(self, x):
        sortedListAccordingToValue = [k for k in sorted(x, key=x.get, reverse=True)]
        sortedListAccordingToKeyLength = sorted(list(x.keys()), key=len, reverse=True)
        topThreeAccordingToKeyLength = sortedListAccordingToKeyLength[:5]
        return topThreeAccordingToKeyLength

    '''Method for extarcting educational requirements'''

    def extractingEducationRequirements(self, df):

        df = df.reset_index(drop=True)
        degreeProgram = self.getStudyProgram()
        programVectorizer = CountVectorizer(ngram_range=(1, 20), token_pattern=r'\b[^\d\W]+\b', min_df=1,
                                            stop_words='english')
        programVectorizer.fit_transform(degreeProgram.cleaned_study_name)
        featureNames = programVectorizer.get_feature_names()

        chunkVectorizer = CountVectorizer(ngram_range=(1, 20), vocabulary=programVectorizer.get_feature_names(),
                                          token_pattern=r'\b[^\d\W]+\b', min_df=1, stop_words='english')
        chunkNgrams = chunkVectorizer.fit_transform(df.cleaned_chunk).toarray()
        chunkNgramsDf = pd.DataFrame(chunkNgrams, columns=chunkVectorizer.get_feature_names())

        allEduReq = []
        for index, row in chunkNgramsDf.iterrows():
            nonZeroData = row.iloc[row.to_numpy().nonzero()[0]]

            educationalRequirements = nonZeroData.to_dict()
            allEduReq.append([educationalRequirements])

        allEduReqDf = pd.DataFrame(allEduReq, columns=['education_requirements'])
        allEduReqDf['top_edu_req'] = allEduReqDf['education_requirements'].apply(lambda x: self.getTopEduReq(x))
        allEduReqDf = pd.concat([df, allEduReqDf], axis=1)
        allEduReqDf.to_csv(r'../data/extracted-data/extracted_education_requirements.csv', index=False)

        return allEduReqDf

    '''-------------------END OF METHODS FOR EXTRACTING EDUCATIONAL REQUIREMENTS INFORMATION-----------------------'''

    '''-------------------START OF METHODS FOR EXTRACTING WORK HOURS INFORMATION-----------------------------------'''
    '''method for work hour extraction ( On Total Job description data )'''

    def extractWorkHours(self, display):

        # df = pd.read_csv("../data/parsed-data/" + self.refinedHtmlFile + ".csv")
        df = pd.read_csv("../data/parsed-data/euro-jobs-parsed.csv")
        # df = df.head(100)

        # print("Cleaning job postings")
        display.set_state(18)
        tqdm.pandas(unit=" job postings")
        df['job_desc_cleaned'] = df['job_desc_plain'].progress_apply(lambda x: self.clean_string(x))
        self._id_index = df.columns.get_loc('id')
        self._col_index = df.columns.get_loc('job_desc_cleaned')
        nlp_d = {}

        # print("Generating features")
        display.set_state(19)
        with tqdm(total=len(df), unit=' job postings') as p_bar:
            for text in df.values.tolist():
                p_bar.update()
                res = self._parallel_nlp(text)
                nlp_d[res[0]] = res[1]
        # with multiprocessing.Pool(processes=os.cpu_count()) as p:
        #     with tqdm(total=len(df), unit=' job postings') as p_bar:
        # for i, res in enumerate(p.imap_unordered(self._parallel_nlp, df.values.tolist())):
        #     p_bar.update()
        #     nlp_d[res[0]] = res[1]
        df['job_desc_doc'] = df['id'].map(nlp_d)
        # df['job_desc_doc'] = df['job_desc_cleaned'].apply(lambda x: nlp(x))

        # print("Extracting work hours")

        display.set_state(20)

        data = []
        for i, row in df.iterrows():

            doc = row['job_desc_doc']
            chunk = row['job_desc_plain']
            cleaned_chunk = row['job_desc_cleaned']

            if self.getOrConditionUsingCombinations(self.workHoursCombinations, cleaned_chunk) or any(
                    x in cleaned_chunk for x in self.workHoursCommonKeywords):

                defaultNormalHours = self.getMatchedElementOfListWithString(self.defaultNormalHoursStings,
                                                                            cleaned_chunk)
                # urationType = self.getMatchedElementOfListWithString(["weekly","monthly"],cleaned_chunk)

                info = []
                for X in doc.ents:
                    if X.label_ in ['TIME'] and X.text not in self.workHoursCommonKeywords and X.text not in info:

                        if bool(re.search(r'\d', X.text)):
                            info.append(X.text)
                        else:
                            word_number_str = self.wordToNumberStr(X.text)
                            if bool(re.search(r'\d', word_number_str)):
                                info.append(word_number_str)

                    elif defaultNormalHours and '40 hours' not in info:
                        info.append('40 hours')

                if info:
                    data.append([row['id'], chunk, cleaned_chunk, info])

        workHoursDf = pd.DataFrame(data, columns=['id', 'job_desc_plain', 'job_desc_cleaned', 'extracted_work_hours'])
        workHoursDf.to_csv(r'../data/extracted-data/category-wise-data/work_hours.csv', index=False)

        return workHoursDf

# ExtractInfo()._init_()


# In[ ]:
