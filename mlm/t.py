from datasets import load_dataset
esg_dataset = load_dataset("csv", data_files='/home/linzhisheng/esg/mlm/source_all_english')
esg_dataset['train'] = esg_dataset['train']
print(esg_dataset['train'])
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
model = AutoModelForMaskedLM.from_pretrained('roberta-large')
tokenizer = AutoTokenizer.from_pretrained('/home/linzhisheng/esg/mlm/roberta-esg-tokenizer')
print("start to tokenize")
def tokenize_function(examples):
    result = tokenizer(examples["Abstract"])
    # if args.mask_stratagy == 'dynamic':
    #     label = tokenizer(examples["Label"])            
    #     result['labels'] = label['input_ids']
    return result
# Use batched=True to activate fast multithreading!
remove_columns = ['Abstract','Label']
# if args.mask_stratagy == 'dynamic':
#     remove_columns.append('Label')
tokenized_datasets = esg_dataset.map(
    tokenize_function, batched=True, remove_columns = remove_columns
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
lm_datasets = tokenized_datasets.map(group_texts, batched=True)
lm_datasets.save_to_disk('lm_datasets_100w')
print(lm_datasets)
