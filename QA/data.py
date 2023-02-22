from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from transformers import AutoModelForQuestionAnswering
import collections
from transformers import TrainingArguments
from transformers import Trainer
import numpy as np
import evaluate
from tqdm.auto import tqdm

# model_checkpoint = "bert-large-uncased-whole-word-masking-finetuned-squad"
# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
n_best = 20
max_answer_length = 30
metric = evaluate.load("squad")

id = 0

def is_chinese(string):
    for ch in string:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False
count = 0
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



raw_datasets = load_dataset('csv', data_files='data/train_fine_grained.csv')
# raw_datasets['train'] = raw_datasets['train'].map(add_question,remove_columns=['value','type','path','text'])


raw_datasets = raw_datasets['train'].train_test_split(test_size=0.2,seed=42)
print(raw_datasets['train'][0])
print(raw_datasets['test'][0])
print(raw_datasets)

# raw_datasets['test'].to_csv('data/test_8_2.csv',index=False)
t = raw_datasets['test'].filter(lambda example:example['text'].find('\t')>=0 or is_chinese(example['text']))
print(t)


