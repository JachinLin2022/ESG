# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
import pdfplumber

class ReportScrapyPipeline:
    def process_item(self, item, spider):
        f = item['content']
        pdf = pdfplumber.open(f)
        content = ''
        count = 0
        for page in pdf.pages:
            content = content + page.extract_text() + '\n'
            count = count + 1
            print(count)
        return item
