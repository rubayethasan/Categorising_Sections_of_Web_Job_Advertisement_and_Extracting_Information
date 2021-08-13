# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html
import pymysql as MySQLdb
from twisted.enterprise import adbapi
from scrapy.utils.project import get_project_settings
settings = get_project_settings()


class EuroJobsPipeline(object):
    def __init__(self):
        dbargs = settings.get('DB_CONNECT')
        db_server = settings.get('DB_SERVER')
        print(db_server)
        dbpool = adbapi.ConnectionPool(db_server, **dbargs)
        self.dbpool = dbpool

    # def __del__(self):
    #     self.dbpool.close()

    # def open_spider(self, spider):
    #     self.__init__()

    def process_item(self, item, spider):
        print('testing url: ' + item['url'])
        # self.dbpool.runQuery("SELECT completion_codes, assigned FROM completion_codes").addCallback(self.receive_result)
        self.dbpool.runInteraction(self.check_select_stmt, item)
        return item

    def check_select_stmt(self, conn, item):
        try:
            result = conn.execute("INSERT INTO JobsEuropeDetails(`URL`, `title`, `date_posted`, `category`, `location`, `job_type`, `description`, `html_blob`, `expired`, `salary`) "
                                  "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                                  (item['url'], item['title'], item['date_posted'], item['category'], item['location'], item['job_type'], item['description'], item['html_blob'], item['expired'], item['salary']))
            if result:
                print("inserted into db")

            print(result)
        except MySQLdb.Error as e:
            print(e)

    def receive_result(self, result):
        print("Receive Result")
        print(result)
        # general purpose method to receive result from defer.
        return result

    def error_query(self, result):
        print("error received", result)
        return result
