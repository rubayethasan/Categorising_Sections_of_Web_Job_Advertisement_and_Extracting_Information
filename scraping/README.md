# euro_jobs_scraper

## Description
1. __requirements.txt__: List of dependent libraries
2. __allowed-country-urls.csv__: A list of urls of the european countries.
3. __ScrapeMain.py__: All the process for scraping are invoked in this file for run the process at once.
4. __euro_jobs/spiders/get_country_wise_job_posting_count.py__: Spider to get country wise job posting listing urls with job count from the portal and populate __country-wise-job-posting.csv__
5. __euro_jobs/spiders/get_job_urls.py__: Spider to get job posting urls from the portal from all its pages and populate __urls.txt__. This file is generated/overwritten when you run scrapy
6. __euro_jobs/spiders/get_job_postings.py__: Spider to get job details for each of the job posting url in __urls.txt__ . The all job postings will be saved in a table named '__myscrapedata__' of a sqlite3 database named '__scrapedata__' . The database file name is __scrapedata.db__ . In this table no duplicate job posting will be saved. The uniqueness of job posting is maintained with url. 
7. __euro_jobs/items.py__: Class to define extracted job fields/features
8. __euro_jobs/pipelines.py__: Class to insert into database
9. __euro_jobs/settings.py__: Class to establish a connection with database

## Install Scrapy
1. [Install scrapy](https://docs.scrapy.org/en/latest/intro/install.html#intro-install)


## Run the process individually

1. To get all the job posting count with assciation of country urls

```scrapy crawl euroJobsCountryWiseJobPostingCount```

2. To get all the job posting urls

```scrapy crawl euroJobsGetJobURLs```

3. To get the job posting details

```scrapy crawl euroJobsGetJobPostingDetails```


## Run all process at once
1. To run all the process above at once run the command below in the terminal
```python ScrapeMain.py```

