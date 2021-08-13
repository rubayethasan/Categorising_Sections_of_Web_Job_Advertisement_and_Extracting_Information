
import sqlite3 as sqlite

# This pipeline takes the Item and stuffs it into scrapedata.db
class EuroJobsPipeline(object):
    def __init__(self):
        self.connection = sqlite.connect('./scrapedata.db')
        self.cursor = self.connection.cursor()
        #self.cursor.execute('DROP TABLE IF EXISTS myscrapedata')
        self.cursor.execute('CREATE TABLE IF NOT EXISTS myscrapedata (id INTEGER PRIMARY KEY, url VARCHAR(255), title VARCHAR(255) ,date_posted VARCHAR(255) ,category VARCHAR(255) ,location VARCHAR(255) ,job_type VARCHAR(255) ,description TEXT ,html_blob TEXT ,expired VARCHAR(255) ,salary VARCHAR(255))')

    # Take the item and put it in database - do not allow duplicates
    def process_item(self, item, spider):
        
        self.cursor.execute("select * from myscrapedata where url=?", (item['url'],)) #checking for duplicate entry in terms of url
        result = self.cursor.fetchone()
        if not result: #not save duplicate url job posting
            self.cursor.execute("insert into myscrapedata (url, title, date_posted, category, location, job_type, description, html_blob, expired, salary) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",(item['url'], item['title'], item['date_posted'], item['category'], item['location'], item['job_type'], item['description'], item['html_blob'], item['expired'], item['salary']))

        self.connection.commit()
        
        return item
                
    def handle_error(self, e):
        log.err(e)