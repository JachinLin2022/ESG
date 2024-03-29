import spacy
from datasets import load_dataset
from datasets import load_from_disk
from datasets import disable_caching
from transformers import AutoTokenizer
import random
tokenizer = AutoTokenizer.from_pretrained('mlm/roberta-esg-tokenizer')
nlp = spacy.load('en_core_web_sm', enable=["tok2vec", "tagger", "parser"])
print(nlp.pipe_names)
disable_caching()
def t(text):
    mask_texts = []
    label_texts = []
    for idx, doc in enumerate(nlp.pipe(text['Abstract'],batch_size=1000)):
        # if idx % 5000 == 0:
        #     print(idx)
        mask_set = set()
        count = 0
        for token in doc:
            # print((token.text, token.head.text, token.dep_, token.tag_))
            # if token.tag_ in ['JJ']:
            #     print('asdas', (token.text, token.dep_, token.head.text, token.tag_))
            if token.dep_ in ['nsubjpass','nsubj', 'dobj', 'amod', 'ROOT'] and token.tag_ not in ['NFP']:
                # print('zzzz', (token.text, token.head.text, token.dep_, token.tag_))
                if token.text.isalpha():
                    mask_set.add((count,token.text))
                    # if count >0 and count < len(doc)-1:
                    #     # print((count,token.text))
                    #     mask_set.add((count-1,doc[count-1].text))
                    #     mask_set.add((count+1,doc[count+1].text))
                # if token.head.text.isalpha():
                #     mask_set.add(token.head.text)
            count = count + 1

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
                input_ids = tokenizer(' ' + token.text)['input_ids']
                if tmp <=8:
                    # print('80%')
                    mask_text = mask_text + ' ' + '<mask>'
                    
                    count_mask = count_mask + 1
                    # print((token.text,input_ids))
                    for i in range(len(input_ids) - 3):
                        # print(token.text)
                        mask_text = mask_text + ' ' + '<mask>'
                        # count_mask = count_mask + 1
                elif tmp == 9:
                    # print('10%')
                    mask_text = mask_text + ' ' + token.text
                else:
                    # print('10% random word')
                    # input_ids = tokenizer(' ' + token.text)['input_ids']
                    if len(input_ids) >= 3:
                        for i in range(len(input_ids) - 2):
                            while 1:
                                ran = random.randint(5,50264)
                                random_token = tokenizer.convert_ids_to_tokens(ran)
                                if random_token.find('Ġ') == 0 and len(random_token) >=2 and len(tokenizer(' ' + random_token[1:])['input_ids']) == 3:
                                    mask_text = mask_text + ' ' + random_token[1:]
                                    break
                    else:
                        # print((len(input_ids),token.text))
                        mask_text = mask_text + ' ' + token.text
                    # for i in range(len(input_ids) - 2):      
                    #     while 1:
                    #         ran = random.randint(0,50264)
                    #         random_token = tokenizer.convert_ids_to_tokens(ran)
                    #         if random_token.find('Ġ') == 0:
                    #             mask_text = mask_text + ' ' + random_token[1:]
                    #             break
                        

            else:
                mask_text = mask_text + ' ' + token.text
            label_test = label_test + ' ' + token.text
            count = count + 1
        mask_texts.append(mask_text)
        label_texts.append(label_test)
    text['Label'] = label_texts
    text['Abstract'] = mask_texts
    return text

def check(text):
    t1 = tokenizer(text['Abstract'])
    t2 = tokenizer(text['Label'])
    if len(t1['input_ids']) != len(t2['input_ids']):
        print(text)
        
def dynamic_mask():
    # data = load_from_disk('C:\\Users\\Jachin\\Downloads\\esg_datasets_all')
    data = load_dataset('csv', data_files='mlm/source_all_english')
    train = data['train'].select(range(1000))
    # print(train[0])
    train = train.map(t,batched=True,batch_size=1000,num_proc=16)
    # train = train.add_column('Mask', train['Label'])
    # train['Mask'][0] = '12312'
    # train['Label'][0] = '412123'
    for i in range(10):
        print(train[i])
    
    # train.map(check,batched=True,batch_size=1000,num_proc=16)
    # print(train[0])
    # train.save_to_disk('dynamic_mask_middle_datasets')
def preprocess():
    from datasets import load_dataset
    esg_dataset = load_from_disk('dynamic_mask_middle_datasets')
    # esg_dataset = esg_dataset.select(range(100000))
    print(esg_dataset)
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('/home/linzhisheng/esg/mlm/roberta-esg-tokenizer')
    print("start to tokenize")
    def tokenize_function(examples):
        result = tokenizer(examples["Abstract"])
        label = tokenizer(examples["Label"])
        # if len(result['input_ids']) != len(label['input_ids']):
        #     return
        result['labels'] = label['input_ids']
        # if args.mask_stratagy == 'dynamic':
        #     label = tokenizer(examples["Label"])            
        #     result['labels'] = label['input_ids']
        return result
    # Use batched=True to activate fast multithreading!
    remove_columns = ['Abstract','Label']
    # if args.mask_stratagy == 'dynamic':
    #     remove_columns.append('Label')
    tokenized_datasets = esg_dataset.map(
        tokenize_function, batched=True, remove_columns = remove_columns,num_proc = 16
    )
    print(tokenized_datasets)
    chunk_size = 128
    def group_texts(examples):
        # Concatenate all texts
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        # Compute length of concatenated texts
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the last chunk if it's smaller than chunk_size
        total_length = (total_length // chunk_size) * chunk_size
        # Split by chunks of max_len
        result = {
            k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
            for k, t in concatenated_examples.items()
        }
        # Create a new labels column
        # if args.mask_stratagy != 'dynamic':
        #     result["labels"] = result["input_ids"].copy()
        return result
    lm_datasets = tokenized_datasets.map(group_texts, batched=True, num_proc=16)
    lm_datasets.save_to_disk('mask_datasets_middle_all')
    print(lm_datasets)
# def split(source)
#     from datasets import load_from_disk
#     downsampled_dataset = t["train"].train_test_split(
#             train_size=1000000, test_size=100000, seed=42
#         )
#     downsampled_dataset.save_to_disk('esg_datasets_all')
#     print(downsampled_dataset)
def main():
    # preprocess()
    dynamic_mask()

if __name__ == '__main__':
    main()