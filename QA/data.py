from datasets import *
from tqdm.auto import tqdm
import pandas as pd
import requests
import io
import PyPDF2


disable_caching()
id = 0
count = 0
def is_chinese(string):
    for ch in string:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False

def test_chinese(example):
    if is_chinese(example['text']) or example['text'].find('\t') >=0:
        global count
        count = count + 1
        return example

def add_question(example):
    global id
    id = id + 1
    example['question'] = "How much {}?".format(example['path'])
    example['context'] = example['text']
    example['answers'] = {
        'text': [example['value']],
        'answer_start': [example['text'].find(example['value'])]
    }
    example['id'] = str(id)
    return example

def split_test():
    raw_datasets = load_dataset('csv', data_files='data/train_fine_grained_url.csv')
    # raw_datasets['train'] = raw_datasets['train'].map(add_question,remove_columns=['value','type','path','text'])


    raw_datasets = raw_datasets['train'].train_test_split(test_size=0.2,seed=42)
    print(raw_datasets['train'][0])
    print(raw_datasets['test'][0])
    print(raw_datasets)

    raw_datasets['test'].to_csv('data/test_8_2_url.csv',index=False)
    # t = raw_datasets['test'].filter(lambda example:example['text'].find('\t')>=0 or is_chinese(example['text']))
    # print(t)
    

    
def request_report(example):
    link = example['url']
    proxies = {
        'http': '127.0.0.1:7890',
        'https': '127.0.0.1:7890',
    }
    try:
        r = requests.get(link)
        print(r)
        if r.status_code == 200:
            global count

            print(url_source.filter(lambda x: x['report'] != ''))
            
            f = io.BytesIO(r.content)
            reader = PyPDF2.PdfReader(f)
            content = ''
            for page in reader.pages:
                content = content + page.extract_text() + '\n'
            # print(len(reader.pages))
            # print(contents)
            example['report'] = content
            return example
        else:
            example['report'] = ''
    except:
        # print('error {}'.format(link))
        example['report'] = ''
    return example

def get_all_test_full_report():
    # url_source = pd.read_csv('data/test_8_2_url.csv',nrows=10)
    # duplicate = url_source.drop_duplicates(subset=['url'],keep='first')
    # print(duplicate)
    # duplicate.map(request_report)
    # print(url_source)
    url_source = load_dataset('csv', data_files='data/test_8_2_url.csv')
    url_source = url_source['train'].select(range(10))
    
    url_source = url_source.map(request_report,num_proc=10)
    print(url_source)
    url_source = url_source.filter(lambda x: x['report'] != '')
    print(url_source)
    url_source = url_source.remove_columns('text')
    url_source.to_csv('report.csv',index=True)

    
def process_report():
    reports = pd.read_csv('/home/linzhisheng/esg/QA/data/report.csv')
    print(reports)
    reports = reports[reports['report'].apply(lambda x: is_chinese(x) == False)]
    reports = reports.rename(columns={'report':'text'})
    print(reports)
    reports.to_csv('data/report_eng.csv')

# split_test()
# get_all_test_full_report()
process_report()