# -*- coding: utf-8 -*-
import scrapy
import re
from Zalando.items import ZalandoItem
import json

URL_BASE = 'https://www.zalando.co.uk/'
REQUEST_BASE = 'https://www.zalando.co.uk/api/catalog/articles?'
#服装类型
#TYPE_BASE = ['mens-clothing-shirts/','mens-clothing-t-shirts/','mens-clothing-jumpers-cardigans/',
			#'mens-sports-clothing/','mens-clothing-jackets/','mens-clothing-jeans/','mens-clothing-trousers-chinos/',
			#'mens-clothing-coats/','mens-clothing-suits-ties',
			#'womens-clothing-blouses-tunics/','womens-clothing-coats/','womens-clothing-jackets/',
			#'womens-clothing-jeans/','womens-clothing-jumpers-cardigans/','womens-sports-clothing/']

class ZalandoSpiderSpider(scrapy.Spider):
	name = 'zalando'
	allowed_domains = ['zalando.co.uk']

	def __init__(self, TYPE_BASE = None, *args, **kwargs):
		super(ZalandoSpiderSpider, self).__init__(*args, **kwargs)
		self.start_urls = [URL_BASE + TYPE_BASE]
		print("Start crawling type: %s" % (TYPE_BASE))

	#解析获得该类型服装所有页码的链接
	def parse(self, response):
		#获取该类型服装总页数(每页若干件服装)
		page_num = response.xpath('//div[@class="cat_label-2W3Y8"]/text()').re(u' \d+')[1]
		print("page num:%s" % (page_num))
		#根据页数，整理出该类型服装其他页码的链接
		for index in range(1,int(page_num) + 1):
		#for index in range(1 , 3): #test!!!!!!!!
			new_url = response.url + str('?p=%d'%index)
			yield scrapy.Request(url = new_url, callback = self.parse1)

	def parse1(self, response):
		#获得服装类型
		# print(response.url)
		TYPE = response.url.split('/')[-2]
		index = response.url.split('/')[-1]
		if index == '':
			index = 0
		else:
			index = int(index.split('=')[-1]) -1
		#print(index)
		requrl = REQUEST_BASE + 'categories=' + TYPE + '&limit=84'
		new_url = requrl + str('&offset=%d' % (index*84)) + '&sort=popularity'
		requests = scrapy.Request(url = new_url, callback = self.parse2)
		requests.meta['TYPE'] = TYPE #传递参数：服装类型
		yield requests

	#动态网页，再次请求
	#从每页获取所有服装url
	def parse2(self, response):
		Info = json.loads(response.text)
		item = ZalandoItem()
		# item['dir_name'] = [x["sku"] for x in Info["articles"]]
		# with open("/home/sky/Zalando/sky.json","a") as f:
		# 	json.dump(item['dir_name'],f)
		# item['outfit_desc'] = Info
		Url_list = [x["url_key"] for x in Info["articles"]]
		# with open("/home/sky/Zalando/sky.json","a") as f:
		# 	json.dump(Url_list,f)
		for url in Url_list:#[0:5]:
			full_url = URL_BASE + url + '.html'
			requests = scrapy.Request(url = full_url, callback = self.parse3)
			requests.meta['TYPE'] = response.meta['TYPE']
			yield requests

	#解析获得该页服装图片及产品描述 
	def parse3(self, response):
		items = []
		item = ZalandoItem()
		TYPE = response.meta['TYPE']
		#该产品网页
		item['link_url'] = response.url
		#保存文件夹名称
		dir_name = response.xpath('//link[@media="only screen and (max-width: 640px)"]/@href').extract()[0].split('-')
		item['dir_name'] = [TYPE + '/' + dir_name[-2] + '-' + dir_name[-1].split('.')[0]]
		#该产品描述
		item['product_desc'] = response.xpath('//h1[@class="h-text h-color-black title-typo h-clamp-2"]/text()').extract()
		#该产品各角度展示图片url
		product_urls = response.xpath('//div[@id="topsection-thumbnail-scroller"]//img/@src').extract()
		Product_imgname = []
		for product in product_urls:
			Product_imgname.append(''.join(product.split('/')[-7:]))
		item['product_imgname'] = Product_imgname
		#配套服装描述
		##由于更新后的zalando网页服装单品超过三个后会翻页，每页不足3个单品时会重复最后一个单品，补全为3的倍数，所以要去重
		desc_temp = response.xpath('//div[contains(@class,"qKyl2a")]//div[@class="oBx8Wu _7lPid_"]//span[contains(@class,"E1U")]/text()').extract()
		outfit_desc = list(set(desc_temp)) #set会按升序排列
		outfit_desc.sort(key = desc_temp.index) #恢复原来的顺序
		item['outfit_desc'] = outfit_desc 
		#配套服装单品图片url
		#同配套服装描述处理方式
		single_temp = response.xpath('//div[contains(@class,"qKyl2a")]//img/@src').extract()
		single_urls = list(set(single_temp)) 
		single_urls.sort(key = single_temp.index) #调整回原来的顺序，使得与服装描述顺序一致
		#outfit_url为整体的那张图片
		outfit_url = response.xpath('//div[contains(@class,"KLJqJJ")]//img/@src').extract()
		if outfit_url != []:
			outfit_url = [outfit_url[0]] #只要整体照片，除去第二个不要的图标
		outfit_urls = outfit_url + single_urls
		Outfit_imgname = []
		for outfit in outfit_urls:
			Outfit_imgname.append(''.join(outfit.split('/')[-7:]))
		item['outfit_imgname'] = Outfit_imgname
		Imgurl_new = []
		img_url = product_urls + outfit_urls
		for url in img_url:
			#修改小图像素，转变为大图url 
			url = url.replace('packshot/pdp-thumb','pdp-zoom') #product_urls
			url = url.replace('pdp-thumb','pdp-zoom')  		   #product_urls
			url = url.replace('pdp-thumb','pdp-zoom') 
			url = url.replace('packshot/pdp-reco-2x','pdp-zoom')  #outfit_urls
			url = re.sub(r'mosaic\d+','mosaic01',url)
			Imgurl_new.append(url)
		#所有需要图片url
		item['image_urls'] = list(set(Imgurl_new))#去重
		items.append(item)
		#返回items，交给item pipeline下载图片
		return items