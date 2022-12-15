# import torch



# t = torch.tensor([[1,2,3]])
# b = torch.tensor([[1,6,3]])
# print(t.dim())
# # t[b] = 555
# print(t==b)

# t[t==b] = -100

# print(t)
import pandas as pd
from transformers import AutoTokenizer
import random
from datasets import load_from_disk
# print('3/2'.isalpha())
model_checkpoint = 'roberta-esg-tokenizer'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
# a = {'paid', 'Issued', 'Ordinary', 'crores', 'Particulars', 'up', 'Rupees', 'Authorised', 'Shares'}

# print(tokenizer(' WBCSD'))
# test = ' I love you'
# r = random.randint(0,len(tokenizer)-1)
# t = tokenizer.decode(r)
# print(t)
# print(tokenizer.encode(t))
# print(tokenizer.encode(test + t))
# print(tokenizer.encode(test))
# print(tokenizer.decode(tokenizer.encode(test + t)))
# exit(0)
# print(tokenizer('    ĠWBCSD    '))
# # print(len(tokenizer('Interpublics Directors')['input_ids']))
# print(tokenizer('    Ġ<mask> <mask> <mask>    ')['input_ids'])
# for k in a:
#     test = tokenizer(k)
#     print(test)
#     print(tokenizer.decode(test['input_ids']))

# r = random.randint(0,len(tokenizer)-1)
# t = tokenizer.decode(r)
# print(t)

# print(tokenizer.encode(t))
# exit(0)


# esg_dataset = pd.read_csv('/home/linzhisheng/esg/mlm/source_mask_80%', nrows=20000)
esg_dataset = load_from_disk('/home/linzhisheng/esg/dynamic_mask_80_10_10_datasets_ttt')


def check(text):
    t1 = tokenizer(text['Abstract'])
    t2 = tokenizer(text['Label'])
    if len(t1['input_ids']) != len(t2['input_ids']):
        print(text)
count = 0
esg_dataset.map(check,batched=False, num_proc=16)


# for idx,row in enumerate(esg_dataset):
#     if idx  % 10000 == 0:
#         print(idx)
#     # print(row['Abstract'])
#     # print(row['Label'])
#     t1 = tokenizer(row['Abstract'])
#     t2 = tokenizer(row['Label'])
#     if len(t1['input_ids']) != len(t2['input_ids']):
#         print(idx)
#         print(t1['input_ids'])
#         print(t2['input_ids'])
#         # print(row['Abstract'])
#         # print(row['Label'])
#         # inter = set(tokenizer.decode(t1['input_ids']).split(' '))&set(tokenizer.decode(t2['input_ids']).split(' '))
        
#         # print(set(tokenizer.decode(t1['input_ids']).split(' ')) - set(inter))
#         # print(set(tokenizer.decode(t2['input_ids']).split(' ')) - set(inter))
#         print(tokenizer.decode(t1['input_ids']))
#         print(tokenizer.decode(t2['input_ids']))
#         print(len(t1['input_ids']))
#         print(len(t2['input_ids']))
#         # break
