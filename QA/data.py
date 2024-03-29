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
    # print(string)
    count = 0
    string = str(string)
    for ch in string:
        if u'\u4e00' <= ch <= u'\u9fff':
            count = count + 1
            if count > 10000:
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
    reports = pd.read_csv('/home/linzhisheng/esg/QA/report_all_new.csv')
    print(reports)
    reports_no_eng = reports[reports['text'].apply(lambda x: is_chinese(x) == True)]
    reports = reports[reports['text'].apply(lambda x: is_chinese(x) == False)]
    # reports = reports.rename(columns={'report':'text'})
    print(reports)
    reports.to_csv('new/report_all_eng.csv')
    reports_no_eng.to_csv('new/report_all_no_eng.csv')
    print(reports_no_eng)
    
def get_train_test_csv():
    train_source = pd.read_csv('/home/linzhisheng/esg/QA/train_fine_grained_index.csv')
    report_valid = pd.read_csv('/home/linzhisheng/esg/QA/new/report_all_figure.csv')
    print(train_source)
    train_data = train_source[train_source['Unnamed: 0'].apply(lambda x: x not in report_valid['Unnamed: 0.1'].values)]
    test_data = train_source[train_source['Unnamed: 0'].apply(lambda x: x in report_valid['Unnamed: 0.1'].values)]
    train_data.to_csv('new/train.csv',index=False)
    test_data.to_csv('new/test.csv',index=False)
    print(train_data)
    print(test_data)
    # print(7 in report_valid['Unnamed: 0.1'].values)

# split_test()
# get_all_test_full_report()
# process_report()
# get_train_test_csv()