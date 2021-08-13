#!/usr/bin/env python
# coding: utf-8

# In[26]:


'''importing libraries'''
import ast
import datetime
import re
import string

import numpy as np
import pandas as pd
import pycountry
from currency_converter import CurrencyConverter
from nltk.corpus import stopwords

stopwords = stopwords.words('english')
from nltk.stem import WordNetLemmatizer
import unicodedata


class ProcessAndMergeInfo:
    
    sourceDataRowCount = 100000
    prirityListEmpType = ['full-time','part-time','internship','freelancing','zero-hour']
    categories = ['education_requirements','employment_type','job_location','work_hours','base_salary','currency_info',
                  #'hiring_organization','responsibilities','description','experience_requirements','qualifications',
                  #'skills','special_commitments','deadline_date','start_date','job_fields','incentive_compensation','job_benefits'
                 ]

    def _init_(self,PARAMETERS, display):

        self._c = CurrencyConverter()

        # csv file list of degree list
        self.degree_list_csv = PARAMETERS['degree_list_csv']
        self.degree_list = self.getDegreeList()
        self.degreePriorityList = list(self.degree_list_csv.keys())
        self.salaryCommonKeywords = list(pd.read_csv(r'../data/essential/salary_related_keywords.csv')['key_word'])


        '''----------------------------------------PROCESSING START-------------------------------------------------'''
        '''un comment those categroies which need to be processed'''
        #KEYWOD TYPE CATEGORIES
        #print(1)
        display.set_state(28)
        self.generateCategoryWiseMergedDataKeywordType('education_requirements','top_edu_req')
        #print(1)
        display.set_state(29)
        self.generateCategoryWiseMergedDataKeywordType('job_location','job_location')
        #print(1)
        display.set_state(30)
        self.generateCategoryWiseMergedDataKeywordType('employment_type','employment_type')
        #print(1)
        display.set_state(31)
        self.generateCategoryWiseMergedDataKeywordType('base_salary','base_salary')
        #print(1)
        display.set_state(32)
        self.generateCategoryWiseMergedDataKeywordType('base_salary','currency_info')
        #print(1)
        #self.generateCategoryWiseMergedDataKeywordType('hiring_organization','hiring_organization')
        #self.generateCategoryWiseMergedDataKeywordType('deadline_date','deadline_date')
        #self.generateCategoryWiseMergedDataKeywordType('start_date','start_date')
        #self.generateCategoryWiseMergedDataKeywordType('work_hours','work_hours')
        
        
        #TEXT TYPE CATEGORIES
        #self.generateCategoryWiseMergedDataTextType('experience_requirements')
        #self.generateCategoryWiseMergedDataTextType('incentive_compensation')
        #self.generateCategoryWiseMergedDataTextType('job_benefits')
        #self.generateCategoryWiseMergedDataTextType('responsibilities')
        #self.generateCategoryWiseMergedDataTextType('special_commitments')
        #self.generateCategoryWiseMergedDataTextType('skills')
        #self.generateCategoryWiseMergedDataTextType('qualifications')
        '''----------------------------------------PROCESSING END---------------------------------------------------'''
        
        
        '''--------------------------------------CLEANNING START---------------------------------------------------'''
        '''un comment those categroies which need to be cleaned'''
        self.cleanJobLocation()
        display.set_state(33)
        self.cleanBaseSalary()
        #print(1)
        display.set_state(34)
        self.cleanEmploymentType()
        #print(1)
        display.set_state(35)
        self.cleanWorkHours()
        #print(1)
        display.set_state(36)
        self.cleanEducationRequirements()
        #print(1)
        '''--------------------------------------CLEANNING END-----------------------------------------------------'''


        '''--------------------------------------MERGING START-----------------------------------------------------'''
        display.set_state(37)
        self.mergeInformation()
        display.set_state(0)
        '''--------------------------------------MERGING END-------------------------------------------------------'''
        

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
            data = data.replace('euro','€')
            data = data.replace('eur','€')
            data = data.replace('dollar','$')
            data = data.replace('usd','$')
            data = re.sub(r"(\d+) (€)", r"€ \g<1>", data) # 2984 € => € 2984
            data = re.sub(r"(\d+)(€)", r"€ \g<1>", data) # 2984€ => € 2984
            data = re.sub(r"(\d+) (\$)", r"$ \g<1>", data) # 2984 $ => $ 2984
            data = re.sub(r"(\d+)(\$)", r"$ \g<1>", data) # 2984$ => $ 2984
            data = re.sub(r"(\d+) (£)", r"£ \g<1>", data) # 2984 € => € 2984
            data = re.sub(r"(\d+)(£)", r"£ \g<1>", data) # 2984€ => € 2984
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
        except:
            return ''
    '''---------------------------------------END OF GENERIC METHODS----------------------------------------------'''
        
        
        
    '''-----------------START OF METHODS FOR PROCESSING KEYWOD TYPE CATEGORY INFORMATION---------------------------'''
    '''method for marging information got from multiple chunks of same data point'''
    def mergeMultipleInfoForSameDataPointKeywordType(self,df,column_name):
        merged_data = []
        for data_point_id in df['data_point_id'].unique():
            rows = df[df['data_point_id'] == data_point_id]
            info = []
            chunk = []
            cleaned_chunk = []
            for i,j in rows.iterrows(): #for merging multiple arrays of information for same datapoint
                saved_info = ast.literal_eval(j[column_name])

                if column_name == 'job_location':
                    info = info + list(set(saved_info))
                elif column_name == 'base_salary':
                    surrounding_words = j['surrounding_words']
                    currency_associated_salary_info = ast.literal_eval(j['currency_associated_salary_info'])
                    if currency_associated_salary_info:
                        info = currency_associated_salary_info
                        chunk.append(j['chunk'])
                        cleaned_chunk.append(j['cleaned_chunk'])
                        break
                    elif list(set(self.salaryCommonKeywords).intersection(surrounding_words.split())):
                        info = info + list(set(saved_info) - set(info))
                else:
                    info = info + list(set(saved_info) - set(info))

                chunk.append(j['chunk'])
                cleaned_chunk.append(j['cleaned_chunk'])

            if info:
                infoLenth = len(info)
                info = map(str, info)
                info = ','.join(info)
                chunk = ','.join(chunk)
                cleaned_chunk = ','.join(cleaned_chunk)
                if column_name == 'base_salary':
                    merged_data.append([data_point_id,chunk,cleaned_chunk,surrounding_words,info,info,infoLenth])
                else:
                    merged_data.append([data_point_id,chunk,cleaned_chunk,info,info,infoLenth])

        if column_name == 'base_salary':
            df_columns = ['id', 'chunks', 'cleaned_chunks','surrounding_words', 'info', 'extracted_' + column_name, 'info_length']
        else:
            df_columns = ['id', 'chunks', 'cleaned_chunks', 'info', 'extracted_' + column_name, 'info_length']

        merged_df = pd.DataFrame(merged_data,columns = df_columns)
        merged_df.sort_values("id", inplace = True)
        merged_df.reset_index(drop=True,inplace = True)
        return merged_df

    '''mathod for genarating amrged data and saving into file'''
    def generateCategoryWiseMergedDataKeywordType(self,category,column_name):
        df = pd.read_csv("../data/extracted-data/extracted_"+category+".csv")
        merged_df =  self.mergeMultipleInfoForSameDataPointKeywordType(df, column_name)
        if column_name == 'currency_info':
            merged_df.to_csv(r'../data/extracted-data/category-wise-data/currency_info.csv', index = False)
        else:
            merged_df.to_csv(r'../data/extracted-data/category-wise-data/'+category+'.csv', index = False)

        return merged_df
    '''-----------------END OF METHODS FOR PROCESSING KEYWOD TYPE CATEGORY INFORMATION-----------------------------'''
    
    
    
    '''-----------------START OF METHODS FOR PROCESSING TEXT TYPE CATEGORY INFORMATION-----------------------------'''
    '''method for marging information got from multiple chunks of same data point'''
    def mergeMultipleInfoForSameDataPointTextType(self,df):
        merged_data = []
        for data_point_id in df['data_point_id'].unique():
            rows = df[df['data_point_id'] == data_point_id]
            info = ''
            for i,j in rows.iterrows():
                info = info + '. ' + self.clean_string(j['chunk'])
            
            info = re.sub(r'^.\s+', '', info) #Removing prefixed 'DOT'
            merged_data.append([data_point_id,info])  
            merged_df = pd.DataFrame(merged_data,columns = ['id','info'])
            merged_df.sort_values("id", inplace = True)
            merged_df = merged_df.reset_index(drop=True)
        return merged_df

    '''mathod for genarating amrged data and saving into file'''
    def generateCategoryWiseMergedDataTextType(self,category):
        df = pd.read_csv("../data/extracted-data/extracted_"+category+".csv")
        merged_df =  self.mergeMultipleInfoForSameDataPointTextType(df[df['category'] == category])
        merged_df.to_csv(r'../data/extracted-data/catergory-wise-data/'+category+'.csv', index = False)
        return merged_df
    
    '''-----------------END OF METHODS FOR PROCESSING TEXT TYPE CATEGORY INFORMATION-------------------------------'''
    
    
    
    '''-----------------START OF METHODS FOR CLEANING BASE SALARY INFORMATION--------------------------------------'''
    '''method for genarating yearly salary'''
    def getYearlySalary(self, row):
        id = row['id']
        estimated_salary = self._estimatedSalary[self._estimatedSalary['id'] == id]['estimated_salary'].values[0]
        currency_info = self._currency_info[self._currency_info['id'] == id]['info'].values
        if len(currency_info) == 1:
            currency_info = currency_info[0]
        else:
            currency_info = None
        job_location = self._job_location[self._job_location['id'] == id]['info'].values
        if len(job_location) == 1:
            job_location = job_location[0]
        else:
            job_location = None

        #print(estimated_salary)
        info = row['extracted_base_salary']
        info = str(info)
        info_str = info
        info = info.split(',')
        info = [float(i) for i in info]
        yearlyInfo = []
        for k in info:

            if currency_info is None or 'euro' not in currency_info:
                detected = False
                # try with detected currency
                if currency_info is not None:
                    try:
                        converted = self._c.convert(k, currency_info)
                        detected = True
                        k = converted
                    except ValueError:
                        pass
                # otherwise try deducting the currency from the country
                if not detected and job_location is not None:
                    country = pycountry.countries.search_fuzzy(job_location)[0]
                    if country is not None:
                        # 2 see if we have that in our country, if we do return
                        currency = pycountry.currencies.get(numeric=country.numeric)
                        if currency is not None:
                            k = self._c.convert(k, currency.alpha_3)
                            detected = True

            if(k <= 1000000): #remove unusually big numbers . les than 1 M
                if not np.isnan(estimated_salary):
                    original = abs(k - estimated_salary)
                    multiplied = abs(k * 12 - estimated_salary)
                    if multiplied < original:
                        k = k * 12
                else:
                    if k <= 10000: #convert monthly salary to yearly salary
                        k = k*12

                yearlyInfo.append(k)

        if len(yearlyInfo) > 1:
            best = yearlyInfo[0]  # TODO find a better way to define this - it will be the default when estimate is nan
            for i in yearlyInfo:
                if abs(i - estimated_salary) < abs(best - estimated_salary):
                    best = i
            yearlyInfo = [best]

        yearlyInfo = map(str, yearlyInfo)
        yearlyInfo = ','.join(yearlyInfo)
        # print(info_str,'=>',yearlyInfo)
        return yearlyInfo

    '''method for cleaning base salary'''
    def cleanBaseSalary(self):
        base_salary = pd.read_csv('../data/extracted-data/category-wise-data/base_salary.csv')
        self._currency_info = pd.read_csv('../data/extracted-data/category-wise-data/currency_info.csv')
        self._job_location = pd.read_csv('../data/extracted-data/category-wise-data/job_location.csv')
        self._estimatedSalary = pd.read_csv('../data/extracted-data/merged/result-andr.csv')
        base_salary['info'] = base_salary.apply(self.getYearlySalary, axis=1)

        #base_salary['info'] = base_salary.apply(lambda x: self.getYearlySalary(x),axis=1)
        base_salary.to_csv('../data/extracted-data/category-wise-data/base_salary.csv')
        return base_salary
    '''-----------------END OF METHODS FOR CLEANING BASE SALARY INFORMATION----------------------------------------'''
    
    
    
    '''-----------------START OF METHODS FOR CLEANING EMPLOYMENT TYPE INFORMATION----------------------------------'''
    '''method for genarating employment type'''
    def prioritizedEmpType(self,info):
        info = re.sub('intership','internship',info)
        priorityDict = {}
        for k in info.split(','):
            priorityDict[k] = self.prirityListEmpType.index(k)

        prioritizedInfo = min(priorityDict, key=priorityDict.get)
        # print(info,'=>',prioritizedInfo)
        return prioritizedInfo
    
    '''method for cleaning employment type'''
    def cleanEmploymentType(self):
        employment_type = pd.read_csv('../data/extracted-data/category-wise-data/employment_type.csv')
        employment_type['info'] = employment_type['extracted_employment_type'].apply(lambda x: self.prioritizedEmpType(x))
        employment_type.to_csv('../data/extracted-data/category-wise-data/employment_type.csv')
        return employment_type
    '''-----------------END OF METHODS FOR CLEANING EMPLOYMENT TYPE INFORMATION------------------------------------'''
    
    
    
    '''-----------------START OF METHODS FOR CLEANING WORK HOURS INFORMATION---------------------------------------'''
    '''get work hours from am pm format'''
    def getWorkTimeFromPattern(self,text,weekly):
        text = text.replace(',','')
        found_res = re.findall(r'(\d+.\d{2}) (\d+.\d{2})|(\d+.\d{2})-(\d+.\d{2})|(\d+.\d{2}) - (\d+.\d{2})|(\d+.\d{2})- (\d+.\d{2})|(\d+.\d{2}) -(\d+.\d{2})|(\d+.\d{2})am (\d+.\d{2})|(\d+.\d{2}) (\d+.\d{2})pm|(\d+.\d{2})am-(\d+.\d{2})|(\d+.\d{2})-(\d+.\d{2})pm|(\d+.\d{2}) am (\d+.\d{2})|(\d+.\d{2}) (\d+.\d{2}) pm|(\d+.\d{2})am - (\d+.\d{2})|(\d+.\d{2}) - (\d+.\d{2})pm|(\d+.\d{2}) am - (\d+.\d{2})|(\d+.\d{2}) - (\d+.\d{2}) pm|(\d+)am-(\d+)|(\d+)-(\d+)pm|(\d+.\d+)am-(\d+)|(\d+.\d+)-(\d+)pm|(\d+)am-(\d+.\d+)|(\d+)-(\d+.\d+)pm|(\d+)am (\d+)|(\d+) (\d+)pm|(\d+.\d+)am (\d+)|(\d+.\d+) (\d+)pm|(\d+)am (\d+.\d+)|(\d+) (\d+.\d+)pm|(\d+)am - (\d+)|(\d+) - (\d+)pm|(\d+) am - (\d+)|(\d+) - (\d+) pm|(\d+) am (\d+)|(\d+) (\d+) pm', text)
        hours = []
        for i in found_res:
            i = list(filter(None, list(i)))
            i = [j.replace('.', ':') for j in i]
            if len(i) == 2:
                time_start = time_end = False

                try:
                    try:
                        time_start = datetime.datetime.strptime(i[0], '%H:%M')
                    except:
                        time_start = datetime.datetime.strptime(i[0], '%H')

                    try:
                        time_end = datetime.datetime.strptime(i[1], '%H:%M')
                    except:
                        time_end = datetime.datetime.strptime(i[1], '%H')
                except:
                    exception = True

                if time_start and time_end:
                    if time_start > time_end:
                        time_end = time_end + datetime.timedelta(hours=12)

                    time_def = time_end - time_start
                    hour_dif = round(time_def.seconds/3600,2)
                    if hour_dif > 0:
                        if weekly: 
                            hour_dif = hour_dif*4

                        hours.append(hour_dif)

        if hours:
            return hours
        else:
            return False
    
    
    '''method for getting matched elements from a string'''
    def getMatchedElementOfListWithString(self,_list,_str,exact_match = False):
        if exact_match:
            _str = _str.split()
        for i in _list:
            if i in _str:
                return i
        return False

    '''method for work hours information from interval pattern'''
    def getWorkHoursFromIntervalPattern(self,_str,weekly):
        dt = re.findall(r'(\d+-\d+)', _str)
        if dt:
            val = []
            for i in dt:
                # workHours = ast.literal_eval(max(i.split('-')))
                workHours = int(max(i.split('-')))
                if weekly:
                    workHours = workHours*4
                
                val.append(workHours)
            return val
        return False
    
    '''get maximum work hour information from multiple'''
    def getMaxWorkHours(self,x):
        if x:
            return max(x)
        return 0

    '''method for getting cleaned work hours'''
    def getCleanedWorkHours(self,info):
        info = ast.literal_eval(info)
        corrected_info = []
        for k in info:
            #try:
            workHoursAmPm = self.getWorkTimeFromPattern(k,True) #get work hours form am pm pattern  
            if workHoursAmPm: #am pm pattern found
                for i in workHoursAmPm:
                    corrected_info.append(i)

            elif self.getMatchedElementOfListWithString(['week','weeks','wk'],k): #for extarcting weekly hours if week or weeks exist in a string

                weeklyWorkHours = self.getWorkHoursFromIntervalPattern(k,True)#extracting weekly work hours from interval pattern eg 2-30 hours/week  
                if weeklyWorkHours:
                    for i in weeklyWorkHours:
                        corrected_info.append(i)
                else:
                    weeklyHours = re.findall(r"([0-9.]*[0-9]+)", k) # e.g '30.5 hours/week 40 hours' 
                    for i in weeklyHours:
                        corrected_info.append(i)

            else:
                workHoursFromIntervalPattern = self.getWorkHoursFromIntervalPattern(k,False) # e.g '1-4 hour'
                if workHoursFromIntervalPattern:
                    for i in workHoursFromIntervalPattern:
                        corrected_info.append(i)

                elif self.getMatchedElementOfListWithString(['hours','hour','hrs'],k):
                    workHours = re.findall(r"([0-9.]*[0-9]+)", k) # e.g '30.5 hours and 40 hours'
                    for i in workHours:
                        corrected_info.append(i)
            #except:continue

        # print('cleaned work hours: ',corrected_info)
        for i in range(len(corrected_info)):
            try:
                corrected_info[i] = float(corrected_info[i])
            except ValueError:
                corrected_info[i] = 0
        # corrected_info = [float(item) for item in corrected_info]
        return [int(item) for item in corrected_info]

    '''method for cleaning work hours'''
    def cleanWorkHours(self):
        work_hours = pd.read_csv('../data/extracted-data/category-wise-data/work_hours.csv')
        work_hours['cleaned_work_hours'] = work_hours['extracted_work_hours'].apply(lambda x: self.getCleanedWorkHours(x) )
        work_hours['info'] = work_hours['cleaned_work_hours'].apply(lambda x: self.getMaxWorkHours(x) )
        work_hours.to_csv(r'../data/extracted-data/category-wise-data/work_hours.csv', index = False)
        return work_hours
    '''-----------------END OF METHODS FOR CLEANING WORK HOURS INFORMATION-----------------------------------------'''
    
    
    
    '''-----------------START OF METHODS FOR CLEANING EDUCATIONAL REQUIREMENTS INFORMATION-------------------------'''

    '''method for mapping educational degree names'''
    def mappingEducationDegrees(self,info):
        info_s = info.split(',')
        mapped_info = []
        for k in info_s:
            for i in self.degree_list:
                if k in self.degree_list[i]:
                    mapped_info.append(i)

        mapped_info = ','.join(mapped_info)
        # print(info,'=>',mapped_info)
        return mapped_info

    '''method for get all degree names'''
    def getDegreeList(self):
        degrees = {}
        for i in self.degree_list_csv:
            individual_degree_list = list(pd.read_csv(self.degree_list_csv[i])['degree'].values)
            degrees.update({i: individual_degree_list})
        return degrees

    '''method for prioratizing degree name information'''
    def getPrioritizedDegree(self,info):
        priorityDict = {}
        info_s=[]
        for k in info.split(','):
            if k in self.degreePriorityList:
                priorityDict[k] = self.degreePriorityList.index(k)
            else:
                info_s.append(k)

        if priorityDict:
            prioritizedInfo = min(priorityDict, key=priorityDict.get)
            info_s.append(prioritizedInfo)

        info_s =','.join(info_s)
        # print(info,'=>',info_s)

        return info_s
    
    
    '''method for cleaning educational requirements'''
    def cleanEducationRequirements(self):
        education_requirements = pd.read_csv('../data/extracted-data/category-wise-data/education_requirements.csv')
        education_requirements['mapped_info'] = education_requirements['extracted_top_edu_req'].apply(lambda x: self.mappingEducationDegrees(x))
        education_requirements['info'] = education_requirements['mapped_info'].apply(lambda x: self.getPrioritizedDegree(x))
        education_requirements.to_csv('../data/extracted-data/category-wise-data/education_requirements.csv')
        return education_requirements
    '''-----------------END OF METHODS FOR CLEANING EDUCATIONAL REQUIREMENTS INFORMATION---------------------------'''
    
    '''-----------------START OF METHODS FOR MERGING ALL INFORMATION-----------------------------------------------'''
    '''method for genarating categroy wise data for marging'''
    def mergeCategoryTypesInfo(self,df,category):
        cat_df = pd.read_csv('../data/extracted-data/category-wise-data/'+category+'.csv')
        for i,j in cat_df.iterrows():
            df.loc[df['id'] == j['id'], [category]] = j['info']

        return df

    def getPrioritizedJobLocation(self, x):
        locations = x.split(',')
        location = max(locations, key=locations.count)
        # print(locations,'===>>',location)
        return location

    def cleanJobLocation(self):
        job_location = pd.read_csv('../data/extracted-data/category-wise-data/job_location.csv')
        job_location['info'] = job_location['extracted_job_location'].apply(lambda x: self.getPrioritizedJobLocation(x))
        job_location.to_csv('../data/extracted-data/category-wise-data/job_location.csv')
        
    '''method for merging all information'''
    def mergeInformation(self):
        id_df = pd.DataFrame(list(range(1, self.sourceDataRowCount)), index=list(range(0, self.sourceDataRowCount-1)), columns=['id'])
        df = pd.DataFrame(np.nan, index=list(range(0, self.sourceDataRowCount)), columns=self.categories)
        df = pd.concat([id_df, df], axis=1)
        
        for category in self.categories:
            # print(category)
            df = self.mergeCategoryTypesInfo(df,category)
        # df.dropna(how='all',subset=self.categories, inplace=True)
        df.dropna(how='all', inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.to_csv(r'../data/extracted-data/merged/final-data-rub.csv', index = False)
        df.to_csv(r'../data/extracted-data/merged/final-data-rub.tsv', sep='\t', index = False)
        return df
    '''-----------------END OF METHODS FOR MERGING ALL INFORMATION-------------------------------------------------'''
        

#ProcessAndMergeInfo()._init_()


# In[ ]:




