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

# print('3/2'.isalpha())
model_checkpoint = 'roberta-large'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
# a = {'paid', 'Issued', 'Ordinary', 'crores', 'Particulars', 'up', 'Rupees', 'Authorised', 'Shares'}

# print(tokenizer(' WBCSD'))
# print(tokenizer('    ĠWBCSD    '))
# # print(len(tokenizer('Interpublics Directors')['input_ids']))
# print(tokenizer('    Ġ<mask> <mask> <mask>    ')['input_ids'])
# for k in a:
#     test = tokenizer(k)
#     print(test)
#     print(tokenizer.decode(test['input_ids']))
# exit(0)


esg_dataset = pd.read_csv('/home/linzhisheng/esg/mlm/source_200w_mask', nrows=1000000)

count = 0
for idx,row in esg_dataset.iterrows():
    if idx % 10000 == 0:
        print(idx)
    # print(row['Abstract'])
    # print(row['Label'])
    t1 = tokenizer(row['Abstract'])
    t2 = tokenizer(row['Label'])
    if len(t1['input_ids']) != len(t2['input_ids']):
        print(idx)
        # print(t1['input_ids'])
        # print(t2['input_ids'])
        # print(row['Abstract'])
        # print(row['Label'])
        print(tokenizer.decode(t1['input_ids']))
        # print(tokenizer.decode(t2['input_ids']))
        print(len(tokenizer(row['Abstract'])['input_ids']))
        print(len(tokenizer(row['Label'])['input_ids']))
        break
