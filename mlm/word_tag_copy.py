import spacy
nlp = spacy.load("en_core_web_sm")
from datasets import load_dataset
import pandas as pd
from transformers import AutoTokenizer
import re
model_checkpoint = 'roberta-large'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


# # esg_dataset = load_dataset("csv", data_files='source_1w')
esg_dataset = pd.read_csv('/home/linzhisheng/esg/mlm/source_all_format', nrows = 500000)
esg_dataset['Label'] = esg_dataset['Abstract']
print(esg_dataset)
idx = 0

def mask_text(t):
    for doc in nlp.pipe(t['Abstract'],n_process=-1):
        mask_set = set()
        count = 0
        for token in doc:
            if token.dep_ in ['ROOT', 'nsubjpass','nsubj', 'dobj', 'amod'] and token.tag_ not in ['NFP']:
                if token.text.isalpha():
                    mask_set.add((count,token.text))
                # if token.head.text.isalpha():
                #     mask_set.add(token.head.text)
            count = count + 1
        mask_text = ''
        label_test = ''
        count = 0
        for token in doc:
            if token.text.find(' ') >= 0:
                # print(idx)
                continue
            if (count, token.text) in mask_set:
                mask_text = mask_text + ' ' + '<mask>'
                input_ids = tokenizer(' ' + token.text)['input_ids']
                # count_mask = count_mask + 1
                for i in range(len(input_ids) - 3):
                    mask_text = mask_text + ' ' + '<mask>'
                    # count_mask = count_mask + 1
            else:
                mask_text = mask_text + ' ' + token.text
            label_test = label_test + ' ' + token.text
            count = count + 1
        global idx
        esg_dataset['Abstract'][idx] = mask_text
        esg_dataset['Label'][idx] = label_test
        # print(mask_set)
        idx = idx + 1
        if idx % 1000 == 0:
            print(idx)

for i in range(1):
    print(f'zzzzzzz: {i}')
    t = esg_dataset['Abstract']
    # print(t)
    mask_text(t)
mask_text(esg_dataset)


esg_dataset.to_csv('source_300w_mask', index=False)
exit(0)
for idx,row in esg_dataset.iterrows():
    if idx % 10 == 0:
        print(idx)
    # continue
    text = row['Abstract']
    text = re.sub(r" +", ' ', text)
    # text = remove_upprintable_chars(text)
    doc = nlp.pipe(text)
    for token in doc:
        print(token)
    break

    mask_set = set()
    count = 0
    for token in doc:
        # print((token.text, token.head.text, token.dep_, token.tag_))
        # if token.tag_ in ['JJ']:
        #     print('asdas', (token.text, token.dep_, token.head.text, token.tag_))
        
        if token.dep_ in ['ROOT', 'nsubjpass','nsubj', 'dobj', 'amod'] and token.tag_ not in ['NFP']:
            # print('zzzz', (token.text, token.head.text, token.dep_, token.tag_))
            if token.text.isalpha():
                mask_set.add((count,token.text))
            if token.head.text.isalpha():
                mask_set.add(token.head.text)
        count = count + 1
        #     mask_text = mask_text + ' ' + '<mask>'
        # else:
        #     mask_text = mask_text + ' ' + token.text
    # for chunk in doc.noun_chunks:

    #     print('bbbbb', (chunk.text, chunk.root.text, chunk.root.dep_,
    #             chunk.root.head.text))
        # if chunk.root.dep_ in ['nsubjpass', 'nsubj']:
        #     print((chunk.text, chunk.root.text, chunk.root.dep_,
        #         chunk.root.head.text))
        
    # for ent in doc.ents:
    #     if ent.label_ in ['ORG']:
    #         print((ent.text, ent.start_char, ent.end_char, ent.label_))

    mask_text = ''
    label_test = ''
    count_mask = 0
    count = 0
    for token in doc:
        if token.text.find(' ') >= 0:
            # print(idx)
            continue
        # print(idx)
        # print((token.text,len(token.text)))
        if (count, token.text) in mask_set:
            mask_text = mask_text + ' ' + '<mask>'
            input_ids = tokenizer(' ' + token.text)['input_ids']
            count_mask = count_mask + 1
            # print((token.text,input_ids))
            for i in range(len(input_ids) - 3):
                # print(token.text)
                mask_text = mask_text + ' ' + '<mask>'
                count_mask = count_mask + 1
        else:
            mask_text = mask_text + ' ' + token.text
        label_test = label_test + ' ' + token.text
        count = count + 1
    # print(mask_set)
    # print(label_test)
    row['Abstract'] = mask_text
    row['Label'] = label_test
    # print(count_mask/len(doc))
    # print(tokenizer(mask_text))
    # print(tokenizer(label_test))
    # print(tokenizer.decode(tokenizer(label_test)['input_ids']))

    # print(mask_text)
    # break

# esg_dataset.to_csv('source_all_mask', index=False)
print(esg_dataset)