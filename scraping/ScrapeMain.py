#!/usr/bin/env python
# coding: utf-8

# In[2]:


import subprocess
import string


def scrape_main():
    subprocess.run("scrapy crawl euroJobsCountryWiseJobPostingCount".split())
    print('Country wise job posting count collection done.')
    subprocess.run("scrapy crawl euroJobsGetJobURLs".split())
    print('All job postings urls collection done')
    subprocess.run("scrapy crawl euroJobsGetJobPostingDetails".split())
    print('All jobposting collection done')


if __name__ == '__main__':
    scrape_main()