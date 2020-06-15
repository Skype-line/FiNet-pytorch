# # -*- coding: utf-8 -*-

# # Define your item pipelines here
# #
# # Don't forget to add your pipeline to the ITEM_PIPELINES setting
# # See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html

# from scrapy.contrib.pipeline.images import ImagesPipeline
# from scrapy.exceptions import DropItem
# from scrapy import Request
# from Zalando import settings
# import requests
# import os

# class ZalandoPipeline(ImagesPipeline):

# 	def get_media_requests(self, item, info):
# 		for image_url in item['image_urls']:
# 			yield Request(image_url)

# 	def item_completed(self, results, item, info):
# 		image_paths = [x['path'] for ok, x in results if ok]
# 		if not image_paths:
# 			raise DropItem("Item contains no images")
# 		img_path_new = []
# 		for image_path in image_paths:
# 			print(image_path)
# 			print(item['dir_name'][0])
# 			img_path_new.append(image_path.replace('full',item['dir_name'][0]))
# 		print(img_path_new)
# 		item['image_paths'] = img_path_new
# 		return item

# 	def process_item(self, item, spider):
# 		if 'image_urls' in item:
# 			images = []
# 			#文件夹名字
# 			dir_path = '%s/%s' % (settings.IMAGES_STORE, item['dir_name'])
# 			#文件夹不存在则创建文件夹
# 			if not os.path.exists(dir_path):
# 				os.makedirs(dir_path)

# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html

from scrapy.pipelines.images import ImagesPipeline
from scrapy.exceptions import DropItem
from scrapy.http import Request
#from Zalando import settings
#import requests
# import hashlib
from scrapy.utils.python import to_bytes
from scrapy.exporters import JsonItemExporter 


class ZalandoPipeline(ImagesPipeline):

	def get_media_requests(self, item, info):
		for image_url in item['image_urls']:
			yield Request(image_url,meta={'item':item})

	def file_path(self, request, response=None, info=None):
		dir_name = request.meta['item']['dir_name'][0]
		url = request.url
		# image_guid = hashlib.sha1(to_bytes(url)).hexdigest()
		path = dir_name + '/' + ''.join(url.split('/')[-7:])
		# print(path)
		return path

	def item_completed(self, results, item, info):
		# print(results)
		image_paths = [x['path'] for ok, x in results if ok]
		if not image_paths:
			raise DropItem("Item contains no images")
		#img_path_new = []
		#for image_path in image_paths:
		#	print(image_path)
		#	print(item['dir_name'][0])
		#	img_path_new.append(image_path.replace('full',item['dir_name'][0]))
		#print(img_path_new)
		item['image_paths'] = image_paths
		return item