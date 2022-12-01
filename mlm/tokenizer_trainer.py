from datasets import load_dataset
from datasets.dataset_dict import DatasetDict
from transformers import AutoTokenizer
old_tokenizer = AutoTokenizer.from_pretrained("roberta-large")
# This can take a few minutes to load, so grab a coffee or tea while you wait!
raw_datasets = load_dataset('csv', data_files="source_all_english")
print(raw_datasets['train'])
# raw_datasets = DatasetDict({"train": raw_datasets})
# print(raw_datasets)
# exit(0)
def get_training_corpus():
    return (
        raw_datasets['train'][i : i + 1000]["Abstract"]
        for i in range(0, len(raw_datasets['train']), 1000)
    )


training_corpus = get_training_corpus()

example = 'Interpublics Directors are elected each year by Interpublics stockholders at the annual meeting of stockholders. Interpublics Corporate Governance Committee recommends nominees to the Board of Directors, and the Board proposes a slate of nominees to the stockholders for election.'

tokens = old_tokenizer.tokenize(example)
tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 50265)
print(tokens)
tokens = tokenizer.tokenize(example)
print(tokens)

tokenizer.save_pretrained("roberta-esg-tokenizer")