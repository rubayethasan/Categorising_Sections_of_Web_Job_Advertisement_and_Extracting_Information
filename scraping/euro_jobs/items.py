# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class EuroJobsItem(scrapy.Item):
    # define the fields for your item here:
    url = scrapy.Field()
    title = scrapy.Field()
    date_posted = scrapy.Field()
    category = scrapy.Field()
    location = scrapy.Field()
    job_type = scrapy.Field()
    description = scrapy.Field()
    html_blob = scrapy.Field()
    expired = scrapy.Field()
    salary = scrapy.Field()
    pass
