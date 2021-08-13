# -*- coding: utf-8 -*-
import scrapy
from scrapy.spiders import Rule
from scrapy.linkextractors import LinkExtractor
from scrapy.http.request import Request
from scrapy.selector import Selector
import xml.dom.minidom
import re
from lxml import etree
import pandas as pd

class EuroJobsCountryWiseJobPostingCount(scrapy.Spider):
    name = 'euroJobsCountryWiseJobPostingCount'
    allowed_domains = ['eurojobs.com']
    allowedCountryUrls = list(pd.read_csv("allowed-country-urls.csv")['url'])
    
    def start_requests(self):
        url = 'https://www.eurojobs.com/browse-by-country/'
        yield Request(url, self.parse)
        
        
    def parse(self, response):
        sel = Selector(response)
        
        data = []
        for anchor in sel.xpath("//table[@id='browse-items']/tr/td/a").extract():
            a = etree.HTML(anchor)
            a_text = a.xpath('//a/text()')[0]
            a_text = a_text.strip()
            a_job_count = int(re.findall(r'\d+', a_text)[0])
            a_ref = a.xpath('//a/@href')[0]
            
            if a_ref in self.allowedCountryUrls: #Only allow the urls of european countries
                data.append([a_ref,a_job_count])
                
        data = pd.DataFrame(data, columns = ['url','job_count'])
        data.to_csv(r'country-wise-job-posting.csv', index = False)


    def print_url(self, response):
        print("URLS are: "+response.url)

   
