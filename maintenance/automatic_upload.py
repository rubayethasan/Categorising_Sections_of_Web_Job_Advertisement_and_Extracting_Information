import pandas as pd
import pymysql
from tqdm import tqdm


def rearrange(date):
    if date is None:
        return date
    return '.'.join(reversed(date.split('.')))


def upload():
    df = pd.read_csv("../data/extracted-data/merged/" + 'merged-result.csv')
    df = df.where(pd.notnull(df), None)
    # df2 = pd.read_csv("../data/euro-jobs/euro-jobs-data.csv")
    # df2 = df2.where(pd.notnull(df2), None)
    # TODO urgent: find corresponding index from df to df2
    # df = df.head(10)
    connection = pymysql.connect(host='173.249.3.120', user='marco', password='marcomysql57', port=3306,
                                 db='jobposting')
    cursor = connection.cursor()
    cursor.execute("DELETE FROM JOBPOSTING_TEMP_BK")
    cursor.execute("INSERT INTO JOBPOSTING_TEMP_BK SELECT * FROM JOBPOSTING_TEMP;")
    cursor.execute("DELETE FROM JOBPOSTING_TEMP")
    connection.commit()
    cols = 'identifier,url,title,skills,industry,educationRequirements,employmentType,jobLocation,workHours,baseSalary,salaryCurrency'  # ,url,datePosted,validThrough'
    args = []
    for i, row in df.iterrows():
        args.append(list(row))
    # for i, row in df2.iterrows():
    #    if i == len(df):
    #        break
    #    args[i].append(row[1])
    #    args[i].append(rearrange(row[3]))
    #    args[i].append(rearrange(row[9]))
    sql = "INSERT INTO JOBPOSTING_TEMP (" + cols + ") VALUES (" + "%s," * (len(cols.split(",")) - 1) + "%s)"
    start = 0
    chunk = 1000
    with tqdm(total=len(df), unit=' job postings') as p_bar:
        while start != len(df) - 1:
            end = min(start + chunk, len(df) - 1)
            cursor.executemany(sql, args[start: end])
            p_bar.update(end - start)
            start = end
            connection.commit()
    cursor.close()
    connection.commit()
    connection.close()


# date posted
# date expired
# url
# job category

if __name__ == '__main__':
    upload()
