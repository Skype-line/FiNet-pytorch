# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class ZalandoItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    link_url = scrapy.Field()
    product_desc = scrapy.Field()
    outfit_desc = scrapy.Field()
    product_imgname = scrapy.Field()
    outfit_imgname = scrapy.Field()
    image_urls = scrapy.Field()
    image_paths = scrapy.Field()
    dir_name = scrapy.Field()