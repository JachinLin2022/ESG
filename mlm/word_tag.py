import spacy
nlp = spacy.load("en_core_web_sm")
from datasets import load_dataset
import pandas as pd
from transformers import AutoTokenizer
import re
import random
model_checkpoint = 'roberta-large'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

output = '/home/linzhisheng/esg/mlm/source_mask_80%'
# esg_dataset = load_dataset("csv", data_files='source_1w')
esg_dataset = pd.read_csv('/home/linzhisheng/esg/mlm/source_all_english', nrows=1)
esg_dataset['Label'] = esg_dataset['Abstract']
print(esg_dataset)


for idx,row in esg_dataset.iterrows():
    if idx % 1000 == 0:
        print(idx)
    if idx != 0 and idx % 500000 == 0:
        print('check point')
        esg_dataset.to_csv(output, index=False)
    #     continue
    text = row['Abstract']
    # text = re.sub(r" +", ' ', text)
    # text = remove_upprintable_chars(text)
    doc = nlp(text)
    mask_set = set()
    count = 0
    print(len(doc))
    for token in doc:
        # print((token.text, token.head.text, token.dep_, token.tag_))
        # if token.tag_ in ['JJ']:
        #     print('asdas', (token.text, token.dep_, token.head.text, token.tag_))
        
        if token.dep_ in ['nsubjpass','nsubj', 'dobj', 'amod'] and token.tag_ not in ['NFP']:
            # print('zzzz', (token.text, token.head.text, token.dep_, token.tag_))
            if token.text.isalpha():
                # mask_set.add((count,token.text))
                if count >0 and count < len(doc):
                    print((count,token.text))
                    mask_set.add((count-1,doc[count-1].text))
                    mask_set.add((count+1,doc[count+1].text))
            # if token.head.text.isalpha():
            #     mask_set.add(token.head.text)
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
    print(mask_set)
    mask_text = ''
    label_test = ''
    count_mask = 0
    count = 0
    for token in doc:
        if token.text.find(' ') >= 0:
            # print(idx)
            count = count + 1
            continue
        # print(idx)
        # print((count, token.text))
        if (count, token.text) in mask_set:
            tmp = random.randint(1,10)
            if tmp <=8:
                # print('80%')
                mask_text = mask_text + ' ' + '<mask>'
                input_ids = tokenizer(' ' + token.text)['input_ids']
                count_mask = count_mask + 1
                # print((token.text,input_ids))
                for i in range(len(input_ids) - 3):
                    # print(token.text)
                    mask_text = mask_text + ' ' + '<mask>'
                    # count_mask = count_mask + 1
            else:
                # print('10%')
                mask_text = mask_text + ' ' + token.text
            # else:
            #     # print('10% random word')
            #     for i in range(len(input_ids) - 2): 
                    
            #     # print(doc[r])
            #         # print(tokenizer.decode(r))
                    
            #         while 1:
            #             r = random.randint(0,len(tokenizer)-1)
            #             t = tokenizer.decode(r)
            #             if len(tokenizer.encode(t)) == 3:
            #                 mask_text = mask_text + tokenizer.decode(r)
            #                 break
                    

        else:
            mask_text = mask_text + ' ' + token.text
        label_test = label_test + ' ' + token.text
        count = count + 1
    # print(label_test)
    # print(mask_text)
    row['Abstract'] = mask_text
    row['Label'] = label_test
    print(mask_text)
    # print(count_mask/len(doc))
    # print(tokenizer(mask_text))
    # print(tokenizer(label_test))
    # print(tokenizer.decode(tokenizer(label_test)['input_ids']))

    # break

# esg_dataset.to_csv(output, index=False)