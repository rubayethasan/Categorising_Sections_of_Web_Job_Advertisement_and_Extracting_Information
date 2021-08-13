import logging
import multiprocessing
import os
import pause
import pymysql
import requests

from datetime import datetime, timedelta
from tqdm import tqdm


logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


HOST = '173.249.3.120'
USER = 'marco'
PASSWORD = 'marcomysql57'


# region SQL queries

Q_TO_CHECK = """
SELECT url
  FROM JOBPOSTING_TEMP 
 WHERE isValid = 1 AND (lastChecked IS NULL OR TIMESTAMPDIFF(MINUTE, lastChecked, NOW()) > %s);
"""

Q_UPDATE_STATUS = """
UPDATE JOBPOSTING_TEMP
   SET isValid = %s, lastChecked = NOW()
 WHERE url = %s;
"""

Q_DELETE_ALL = """
DELETE FROM JOBPOSTING;
"""

Q_INSERT = """
INSERT INTO JOBPOSTING
SELECT baseSalary,
       datePosted,
       educationRequirements,
       employmentType,
       estimatedSalary,
       experienceRequirements,
       hiringOrganization,
       incentiveCompensation,
       industry,
       jobBenefits,
       jobLocation,
       occupationalCategory,
       qualifications,
       relevantOccupation,
       responsibilities,
       salaryCurrency,
       skills,
       specialCommitments,
       title,
       validThrough,
       workHours,
       identifier,
       url,
       jobCategory,
       openPositions,
       jobIdentifier
FROM JOBPOSTING_TEMP
WHERE title IS NOT NULL AND jobLocation IS NOT NULL
"""

Q_UPDATE_B_SALARY = """
UPDATE JOBPOSTING
SET baseSalary = baseSalary / 12
WHERE baseSalary > 150000;
"""

Q_UPDATE_E_SALARY = """
UPDATE JOBPOSTING
SET estimatedSalary = estimatedSalary / 12
WHERE estimatedSalary > 150000;
"""

Q_FLOOR_SALARIES = """
UPDATE JOBPOSTING
SET baseSalary = FLOOR(baseSalary), estimatedSalary = FLOOR(estimatedSalary);
"""

Q_UPDATE_EMP_T = """
UPDATE JOBPOSTING
SET employmentType = 'full-time'
WHERE employmentType is NULL;
"""

# endregion


def is_valid(url):
    try:
        if requests.get(url).status_code == 200:
            return 1, url
        else:
            return 0, url
    except requests.exceptions.ConnectionError:
        return -1


def update_valid_status(frequency):
    connection = pymysql.connect(host=HOST, user=USER, password=PASSWORD, port=3306, db='jobposting')
    cursor = connection.cursor()

    cursor.execute(Q_TO_CHECK, (frequency * 60,))
    to_check = cursor.fetchall()
    to_check = set([url for t in to_check for url in t])

    if len(to_check) == 0:
        return False

    logging.info("Updating {} entries".format(len(to_check)))

    with multiprocessing.Pool(processes=os.cpu_count()) as p:
        with tqdm(total=len(to_check), unit=' job postings') as p_bar:
            for i, res in enumerate(p.imap_unordered(is_valid, to_check)):
                p_bar.update()
                if res != -1:
                    cursor.execute(Q_UPDATE_STATUS, res)
                    connection.commit()

    cursor.close()
    connection.close()

    return True


def update_displayed_postings(valid=True):
    connection = pymysql.connect(host=HOST, user=USER, password=PASSWORD, port=3306, db='jobposting')
    cursor = connection.cursor()

    cursor.execute(Q_DELETE_ALL)

    if valid:
        cursor.execute(Q_INSERT + "AND isValid = 1 AND lastChecked IS NOT NULL")
    else:
        cursor.execute(Q_INSERT)

    cursor.execute(Q_UPDATE_B_SALARY)
    cursor.execute(Q_UPDATE_E_SALARY)
    cursor.execute(Q_FLOOR_SALARIES)
    cursor.execute(Q_UPDATE_EMP_T)

    connection.commit()
    cursor.close()
    connection.close()

    logging.info("Updated displayed data\n")


if __name__ == '__main__':

    while True:

        if update_valid_status(frequency=12):
            update_displayed_postings(True)

        else:
            next_up = datetime.now() + timedelta(hours=1)
            pause.until(datetime(year=next_up.year, month=next_up.month, day=next_up.day, hour=next_up.hour))
