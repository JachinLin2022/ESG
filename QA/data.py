from datasets import *
from tqdm.auto import tqdm
import pandas as pd
import requests


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
    
    try:
        print(link)
        x = requests.get(link)
        if x.status_code == 200:
            example['report'] = x.text
            return example
    except:
        print(link)
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
    url_source = url_source.map(request_report)
    print(url_source)
    url_source = url_source.remove_columns('text')
    url_source.to_csv('report.csv',index=False)

    



# split_test()
get_all_test_full_report()