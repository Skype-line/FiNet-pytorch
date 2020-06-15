#!/bin/bash
#服装类型
#TYPE_BASE = ['mens-clothing-shirts/','mens-clothing-t-shirts/','mens-clothing-jumpers-cardigans/','mens-sports-clothing/','mens-clothing-jackets/','mens-clothing-jeans/','mens-clothing-trousers-chinos/','mens-clothing-coats/','mens-clothing-suits-ties','womens-clothing-blouses-tunics/','womens-clothing-coats/','womens-clothing-jackets/','womens-clothing-jeans/','womens-clothing-jumpers-cardigans/','womens-sports-clothing/']

scrapy crawl zalando -a TYPE_BASE="mens-clothing-shirts/"                 	\
					 -o Zalando_Info/mens-clothing-shirts.json            	\      
     				 -s LOG_FILE=all.log
scrapy crawl zalando -a TYPE_BASE="mens-clothing-t-shirts/"               	\         
     			     -o Zalando_Info/mens-clothing-t-shirts.json          	\  
     			     -s LOG_FILE=all.log
scrapy crawl zalando -a TYPE_BASE="mens-clothing-jumpers-cardigans/"      	\
					 -o Zalando_Info/mens-clothing-jumpers-cardigans.json 	\
					 -s LOG_FILE=all.log
scrapy crawl zalando -a TYPE_BASE="mens-sports-clothing/"                 	\
					 -o Zalando_Info/mens-sports-clothing.json            	\
					 -s LOG_FILE=all.log
scrapy crawl zalando -a TYPE_BASE="mens-clothing-jackets/"                	\
					 -o Zalando_Info/mens-clothing-jackets.json           	\
					 -s LOG_FILE=all.log 
scrapy crawl zalando -a TYPE_BASE="mens-clothing-jeans/"                  	\
					 -o Zalando_Info/mens-clothing-jeans.json            	\
					 -s LOG_FILE=all.log
scrapy crawl zalando -a TYPE_BASE="mens-clothing-trousers-chinos/"        	\
					 -o Zalando_Info/mens-clothing-trousers-chinos.json   	\
					 -s LOG_FILE=all.log
scrapy crawl zalando -a TYPE_BASE="mens-clothing-coats/"                 	\
					 -o Zalando_Info/mens-clothing-coats.json             	\
					 -s LOG_FILE=all.log
scrapy crawl zalando -a TYPE_BASE="mens-clothing-suits-ties/"             	\
					 -o Zalando_Info/mens-clothing-suits-ties.json        	\
					 -s LOG_FILE=all.log
scrapy crawl zalando -a TYPE_BASE="womens-clothing-blouses-tunics/"       	\
					 -o Zalando_Info/womens-clothing-blouses-tunics.json 	\
					 -s LOG_FILE=all.log
scrapy crawl zalando -a TYPE_BASE="womens-clothing-coats/"               	\
					 -o Zalando_Info/womens-clothing-coats.json          	\
					 -s LOG_FILE=all.log
scrapy crawl zalando -a TYPE_BASE="womens-clothing-jackets/"             	\
					 -o Zalando_Info/womens-clothing-jackets.json         	\
					 -s LOG_FILE=all.log
scrapy crawl zalando -a TYPE_BASE="womens-clothing-jeans/"           	    \
					 -o Zalando_Info/womens-clothing-jeans.json      	    \
					 -s LOG_FILE=all.log
scrapy crawl zalando -a TYPE_BASE="womens-clothing-jumpers-cardigans/"      \
					 -o Zalando_Info/womens-clothing-jumpers-cardigans.json \
					 -s LOG_FILE=all.log
scrapy crawl zalando -a TYPE_BASE="womens-sports-clothing/"                 \
 					 -o Zalando_Info/womens-sports-clothing.json            \
 					 -s LOG_FILE=all.log