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

model_checkpoint = "/home/linzhisheng/esg/QA/esg-QA-deberta"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
n_best = 20
max_answer_length = 30
metric = evaluate.load("squad")


def is_chinese(string):
    for ch in string:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False

id = 0
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


raw_datasets = load_dataset('csv', data_files='/home/linzhisheng/esg/QA/new/train.csv')
test_datasets = load_dataset('csv', data_files='/home/linzhisheng/esg/QA/new/test.csv')
raw_datasets['train'] = raw_datasets['train'].map(add_question,remove_columns=raw_datasets['train'].column_names)
test_datasets['test'] = test_datasets['train'].map(add_question,remove_columns=test_datasets['train'].column_names)
raw_datasets['test'] = test_datasets['test']

print(raw_datasets['train'][0])
print(raw_datasets['test'][0])

max_length = 384
stride = 128


def preprocess_training_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

def preprocess_validation_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs

# train_dataset = raw_datasets["train"].map(
#     preprocess_training_examples,
#     batched=True,
#     remove_columns=raw_datasets["train"].column_names,
# )
# print(len(raw_datasets["train"]), len(train_dataset))
# def to_string(example):
#     example['id'] = str(example['id'])
#     return example
# from datasets import Value
# raw_datasets = load_dataset('csv', data_files='data/test.csv')
# raw_datasets['test'] = raw_datasets['train'].select(range(100))
# print(raw_datasets['test'].features)
# new_features = raw_datasets['test'].features.copy()
# new_features['id'] = Value('string')
# # raw_datasets['test'] = raw_datasets['test'].map(to_string)
# # raw_datasets['test']
# raw_datasets['test'] = raw_datasets['test'].cast(new_features)
# print(raw_datasets['test'][0])


validation_dataset = raw_datasets["test"].map(
    preprocess_validation_examples,
    batched=True,
    remove_columns=raw_datasets["test"].column_names,
)
print(len(raw_datasets["test"]), len(validation_dataset))
filter_list = []
def compute_metrics(start_logits, end_logits, features, examples):
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    # print(theoretical_answers)
    # print(predicted_answers)
    count = 0
    
    for i in range(len(theoretical_answers)):
        want = theoretical_answers[i]['answers']['text'][0]
        get = predicted_answers[i]['prediction_text']
        if get.find(want) >= 0:
            count = count + 1
        # if theoretical_answers[i]['answers']['text'][0] != predicted_answers[i]['prediction_text']:
            # print((i, theoretical_answers[i]['answers']['text'][0], predicted_answers[i]['prediction_text']))
            # print(theoretical_answers[i])
            # count = count + 1
            # filter_list.append(str(i+1))
            # print('\n')
    print(count/len(theoretical_answers))
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)

model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

args = TrainingArguments(
    "bert-finetuned-esgQA-eval",
    per_device_eval_batch_size=128,
    fp16=True,

)
trainer = Trainer(
    model=model,
    args=args,
    # train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
    
)
# trainer.train()
# trainer.save_model()
print(raw_datasets["test"])
print(validation_dataset)
import time
T1 = time.time()
predictions, _, _ = trainer.predict(validation_dataset)
start_logits, end_logits = predictions
print(compute_metrics(start_logits, end_logits, validation_dataset, raw_datasets["test"]))
T2 = time.time()
print('程序运行时间:%s毫秒' % ((T2 - T1)))
# print(raw_datasets["test"].filter(lambda example:example['id'] in filter_list))


