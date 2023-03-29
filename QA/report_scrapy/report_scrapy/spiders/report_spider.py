from pathlib import Path
from datasets import *
import scrapy
import io
import PyPDF2
import pandas as pd
import pdfplumber
from pypdf import PdfReader
from pdfminer.high_level import extract_text
from tika import parser
import fitz
from ..items import ReportScrapyItem
import os
filenames = []
def is_chinese(string):
    for ch in string:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False


def traverse_directory(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            t = os.path.join(root, file)
            filenames.append(t[t.find('report/') + 7:-4])
            
class QuotesSpider(scrapy.Spider):
    name = "report"
    count = 0
    # url_source = load_dataset('csv', data_files='/home/linzhisheng/esg/QA/data/test_8_2_url.csv')
    # url_source = url_source['train'].select(range(100))
    # url_source = url_source.remove_columns('text')
    # url_source = url_source.add_column('report',['']*len(url_source))
    # url_source = pd.read_csv('/home/linzhisheng/esg/QA/data/train_fine_grained_url.csv',nrows=10)
    url_source = pd.read_csv('/home/linzhisheng/esg/QA/new/test.csv')
    print(url_source)
    url_source = url_source.drop(labels='text',axis=1)
    url_source['text'] = ''
    
    path = "/home/linzhisheng/esg/QA/report"
    traverse_directory(path)
    
    print(len(filenames))
    
    
    def start_requests(self):
        # print(self.url_source)
        # yield scrapy.Request(url='https://www.nipponsanso-hd.co.jp/Portals/0/images/ir/library/integrated_report/nippon-sanso-holdings-integrated-report_en-full_2021.pdf', callback=self.parse,meta={'index':'1'})
        for i in range(len(self.url_source)):
            if str(self.url_source['Unnamed: 0'][i]) not in filenames:
                # print(self.url_source['Unnamed: 0'][i])
                yield scrapy.Request(url=self.url_source['url'][i], callback=self.parse,meta={'index':i})
        # for url in urls:
        #     yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        id = response.meta['index']
        print((self.url_source['Unnamed: 0'][id], response.status))
        try:
            with open('/home/linzhisheng/esg/QA/report/{}.pdf'.format(self.url_source['Unnamed: 0'][id]), 'wb') as f:
                f.write(response.body)
            # f = io.BytesIO(response.body)
            # item = ReportScrapyItem()
            # item['content'] = f
            # yield item
            # pdf = pdfplumber.open(f)
            # content = ''
            # count = 0
            # # doc = fitz.open(stream=f, filetype="pdf")  # open document
            # for page in pdf.pages:
            #     content = content + page.extract_text() + '\n'
            #     count = count + 1
            #     print(count)
            # content = extract_text(f)
            # for page in doc:  # iterate the document pages
            #     text = page.get_text()  # get plain text (is in UTF-8)
            #     content = content + text + '\n'
                
                # print(text)
                # out.write(text)  # write text of page
                # out.write(bytes((12,)))  # write page delimiter (form feed 0x0C)
            # print(content)
            
            # reader = PdfReader(f)
            
            # content = ''
            # for page in reader.pages:
            #     content = content + page.extract_text() + '\n'
            #     count = count + 1
            #     print(count)
            # print(id)
            # f = open('f.txt','w')
            # f.write(content)
            # self.url_source['text'][id] = content
            
            # print(self.url_source[id]['report'])
            # print(id)
        except Exception as e:
            print(e)
    
    def closed(self,reason):
        self.url_source = self.url_source[self.url_source['text'] != '']
        print(self.url_source['url'])
        # self.url_source = self.url_source[self.url_source['text'].apply(lambda x: is_chinese(x) == False)]
        # print(self.url_source['url'])
        # self.url_source.to_csv('/home/linzhisheng/esg/QA/report_all_format.csv')
        print('end')