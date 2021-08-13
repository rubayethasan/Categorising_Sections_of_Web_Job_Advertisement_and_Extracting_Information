#!/usr/bin/env python
# coding: utf-8

# In[14]:


'''improting libraries'''
import pandas as pd
import pymysql
from tqdm import tqdm

HOST = '173.249.3.120'
USER = 'marco'
PASSWORD = 'marcomysql57'

MAIN_TABLE = 'JOBPOSTING_TEMP'
BACKUP_TABLE = 'JOBPOSTING_TEMP_BK'

class MergeAndExportInfo:
    url = "../data/extracted-data/merged/"

    def _init_(self, display):
        display.set_state(39)
        self.mergeAll()
        display.set_state(40)
        self._upload_data()  # saveDataIntoSql()
        display.set_state(0)
        # self.exportFinalData()
        # display.set_state(0)

    def merge(self):
        b = pd.read_csv(self.url + "acc.csv", sep='\t', dtype={'id': int,
                                                               'education_requirements': str,
                                                               'employment_type': str,
                                                               'job_location': str,
                                                               'work_hours': pd.Int64Dtype(),
                                                               'base_salary': float,
                                                               'currency_info': str
                                                               })
        a = pd.read_csv(self.url + "result-andr.csv")
        merged = a.merge(b, on='id')
        merged.to_csv(self.url + "output.csv", index=False)

    def cl(self, string):
        if string is None:
            return ''
        return string.replace('[', '').replace(']', '').replace("', ", '!').replace("'", '').replace('¬', ';').replace(
            'n.a.', '').replace('!', ', ').replace('full time', 'full-time').replace('part time', 'part-time')

    def rclean(self, csv, out):
        l = ''
        with open(csv, 'r') as file:
            for line in file.readlines():
                spl = line.split('\t')
                k = 0
                for s in spl:
                    s = s.split(',')[0].replace('\n', '')
                    l += s
                    l += '\t'
                    k += 1
                l = l[:-1]
                l += '\n'
        with open(out, 'w') as file:
            file.writelines(l)

    def clean(self, csv, out):
        l = ''
        with open(csv, 'r') as file:
            for line in file.readlines():
                l += self.cl(line)
        with open(out, 'w') as file:
            file.writelines(l)

    def mergeAll(self):
        self.rclean(self.url + 'final-data-rub.tsv', self.url + 'acc.csv')
        self.merge()
        self.clean(self.url + 'output.csv', self.url + 'merged-result.csv')

    def _upload_data(self):
        df = pd.read_csv("../data/extracted-data/merged/" + 'merged-result.csv')
        df = df.where(pd.notnull(df), None)
        # df2 = pd.read_csv("../data/euro-jobs/euro-jobs-data.csv")
        # df2 = df2.where(pd.notnull(df2), None)
        # TODO urgent: find corresponding index from df to df2
        connection = pymysql.connect(host=HOST, user=USER, password=PASSWORD, port=3306, db='jobposting')
        cursor = connection.cursor()
        cursor.execute("DELETE FROM %s" % BACKUP_TABLE)
        cursor.execute("INSERT INTO %s SELECT * FROM %s;" % (BACKUP_TABLE, MAIN_TABLE))
        cursor.execute("DELETE FROM %s" % MAIN_TABLE)
        cols = 'identifier,url,title,skills,industry,estimatedSalary,educationRequirements,employmentType,jobLocation,workHours,baseSalary,salaryCurrency'  # ,url,datePosted,validThrough'
        args = []
        for i, row in df.iterrows():
            args.append(list(row))
        # for i, row in df2.iterrows():
        #    if i == len(df):
        #        break
        #    args[i].append(row[1])
        #    args[i].append(rearrange(row[3]))
        #    args[i].append(rearrange(row[9]))
        sql = "INSERT INTO %s (" % MAIN_TABLE + cols + ") VALUES (" + "%s," * (len(cols.split(",")) - 1) + "%s)"
        start = 0
        chunk = 1000
        with tqdm(total=len(df), unit=' job postings') as p_bar:
            while start != len(df):
                end = min(start + chunk, len(df))
                cursor.executemany(sql, args[start: end])
                p_bar.update(end - start)
                start = end
        cursor.close()
        connection.commit()
        connection.close()

    def saveDataIntoSql(self):

        df = pd.read_csv(self.url + 'merged-result.csv')
        df = df.where(pd.notnull(df), None)  # python NaN to sql None for avoiding sql insertion error

        # buildingconnection
        connection = pymysql.connect(host='localhost', user='andrea', password='sqlpwd', port=3306, db='diskow_data')
        cursor = connection.cursor()

        if cursor.execute("SHOW TABLES LIKE 'marged_table'"):  # drop previous table if exist
            cursor.execute("DROP TABLE marged_table")

        # create new table
        tableCreateSql = "CREATE TABLE marged_table (id INT(11), url VARCHAR(500), job_title VARCHAR(500), skills VARCHAR(500), sector VARCHAR(500),education_requirements VARCHAR(500), employment_type VARCHAR(500),job_location VARCHAR(500), work_hours INT(11), base_salary VARCHAR(500), currency_info VARCHAR(500))"
        cursor.execute(tableCreateSql)

        # creating column list for insertion
        cols = ",".join([str(i) for i in df.columns.tolist()])

        # Insert DataFrame records one by one.
        with tqdm(total=len(df), unit=' chunks') as p_bar:
            for i, row in df.iterrows():
                p_bar.update()
                # print(row['id'])
                # form query
                sql = "INSERT INTO marged_table (" + cols + ") VALUES (" + "%s," * (len(row) - 1) + "%s)"

                # execute query
                cursor.execute(sql, tuple(row))

                # the connection is not autocommitted by default, so we must commit to save our changes
                connection.commit()

        connection.close()

    def exportFinalData(self):

        connection = pymysql.connect(host='localhost', user='andrea', password='sqlpwd', port=3306, db='diskow_data')
        cursor = connection.cursor()

        cursor.execute('SELECT * FROM marged_table')

        rows = cursor.fetchall()

        field_names = [i[0] for i in cursor.description]

        # print(field_names)

        df = pd.DataFrame(rows, columns=field_names)

        df.to_csv(self.url + 'merged-result.csv', index=False)
        df.to_csv(self.url + 'merged-result.tsv', sep='\t', index=False)

# MergeAndExportInfo()._init_()


# In[ ]:
