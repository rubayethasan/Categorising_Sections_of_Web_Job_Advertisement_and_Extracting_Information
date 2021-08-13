# -*- coding: utf-8 -*-
import scrapy
from scrapy.spiders import Rule
from scrapy.linkextractors import LinkExtractor
from scrapy.http.request import Request
from scrapy.selector import Selector
import pandas as pd
import math


class EurojobsSpider(scrapy.Spider):
    name = 'euroJobsGetJobURLs'
    allowed_domains = ['eurojobs.com']
    maximumJobForEachCountry = 10000 #maximum 10k jobs will be collected for each country
                
    def start_requests(self):
        
        #countryJob = pd.read_csv("country-wise-job-posting.csv",sep=';')
        countryJob = pd.read_csv("country-wise-job-posting.csv",sep=',')

        
        #iterrating over already saved jkb country urls and job count
        for index,row in countryJob.iterrows():
            url = row['url']
            job_count = row['job_count']
            if job_count > self.maximumJobForEachCountry:
                job_count = 10000
            
            #Calculating total page number depending on total job posting cout. 10 jobs in one page.
            #remPage = 0
            #if job_count%10 > 0: remPage= 1
            #totalPageCount = round(job_count/10)+ remPage
            #totalPageCount = math.floor(job_count/10)+ remPage
            totalPageCount = math.ceil(job_count/10)
            
            print('totalJobCount: ',job_count,' totalPageCount: ',totalPageCount,' mainCountryUrl: ',url)
            
            #iterrating over page numbers for generating listing urls for scarpping
            i = 1
            print('totalPageCount:',totalPageCount)
            while i <= totalPageCount:
                #print('i:',i)
                #print('currentPageNumber: ',i)
                urlForScrp = url+'?searchId=1606174454.9407&action=search&page='+str(i)+'&view=list'
                i = i + 1
                yield Request(urlForScrp, self.parse)
        
        
            
    def parse(self, response):
        sel = Selector(response)
        f = open("urls.txt", "a")
        
        #extarct all href for individual job posting
        for href in sel.xpath("//li[@class='viewDetails']/a/@href").extract():
            #print('href',href)
            f.write(href+"\n")
        f.close()
      
    def print_url(self, response):
        print("URLS are: "+response.url)
