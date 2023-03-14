import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering
from transformers import TrainingArguments
from transformers import Trainer
import torch
from datasets import *
import re
import collections
import numpy as np
import heapq
import json

from multiprocessing import Process

class ESG():
    model = 0
    tokenizer = 0
    checkpoint = 0
    trainer = 0
    example_to_features = 0
    start_logits = 0
    end_logits = 0
    n_best = 20
    max_answer_length = 30
    top_n = 20
    # predicted_answers = []
    
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.args = TrainingArguments(
            "main_test",
            evaluation_strategy="no",
            save_strategy="epoch",
            learning_rate=2e-5,
            num_train_epochs=3,
            weight_decay=0.01,
            per_device_train_batch_size=6,
            per_device_eval_batch_size=1250,
            fp16=True,
            # no_cuda=True,
            push_to_hub=False,
            save_total_limit=1
        )
        self.trainer = Trainer(
            model=self.model,
            args=self.args,
            # train_dataset=train_dataset,
            # eval_dataset=eval_set,
            tokenizer=self.tokenizer,

        )
        print('[model init success]')

    def update_model(self, checkpoint):
        print('=============[update_model]==============')
        self.model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)
        self.trainer = Trainer(
            model=self.model,
            args=self.args,
            # train_dataset=train_dataset,
            # eval_dataset=eval_set,
            tokenizer=self.tokenizer,

        )
    
    
    def get_model_result(self, example):
        # print('=============[get_model_result]==============')
        example_id = example["id"]
        context = example["context"]
        answers = []
        score = []
        
        for feature_index in self.example_to_features[example_id]:
            # print((feature_index,tokenizer.decode(eval_set["input_ids"][feature_index])))
            start_logit = self.start_logits[feature_index]
            end_logit = self.end_logits[feature_index]
            offsets = self.offset_mapping[feature_index]

            start_indexes = np.argsort(
                start_logit)[-1: -self.n_best - 1: -1].tolist()
            end_indexes = np.argsort(end_logit)[-1: -self.n_best - 1: -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length.
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > self.max_answer_length
                    ):
                        continue
                    ans = context[offsets[start_index][0]: offsets[end_index][1]]
                    if re.search(r'\d', ans):
                        score.append(start_logit[start_index] + end_logit[end_index])
                        answers.append(
                            {
                                # "id": len(answers),
                                "answer": ans,
                                "logit_score": start_logit[start_index] + end_logit[end_index],
                                # "context": tokenizer.decode(self.eval_set["input_ids"][feature_index], skip_special_tokens=False),
                                # "origin_text": tokenizer.decode(eval_set["input_ids"][feature_index])
                            }
                        )

                    # print((context[offsets[start_index][0] : offsets[end_index][1]], start_logit[start_index] + end_logit[end_index]))

        # best_answer = max(answers, key=lambda x: x["logit_score"])
        from torch.nn.functional import softmax
        score = torch.FloatTensor(score)
        prob = softmax(score,dim=-1)
        for i in range(len(answers)):
            answers[i]['prob'] = prob[i].item()
            
        top_20_answers = heapq.nlargest(
            self.top_n, answers, key=lambda s: s['prob'])
        # predicted_answers.append(
        #     {"id": example_id, "prediction_text": best_answer["text"]})
        example['answer_set'] = top_20_answers
        return example
        # print(top_20_answers)
        
    def get_fiture_set(self, small_eval_set, eval_set, top_n=20):
        print('=============[get_fiture_set]==============')
        self.top_n = top_n
        self.eval_set = eval_set
        predictions, _, _ = self.trainer.predict(eval_set)
        self.start_logits, self.end_logits = predictions
        self.offset_mapping = self.eval_set["offset_mapping"]
        self.example_to_features = collections.defaultdict(list)
        for idx, feature in enumerate(eval_set):
            self.example_to_features[feature["example_id"]].append(idx)
        print('=============[get_model_result]==============')
        firuge_set = small_eval_set.map(self.get_model_result, batched=False)
        return firuge_set
        
        


model_checkpoint = "/home/linzhisheng/esg/QA/esg-QA"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

max_length = 128
stride = 32

id = 0


def add_question_to_get_fiture(example):
    global id
    id = id + 1
    example['question'] = "How much {}?".format(str(example['path']))
    example['context'] = str(example['text'])
    # example['answers'] = {
    #     'text': [example['value']],
    #     'answer_start': [example['text'].find(example['value'])]
    # }
    example['id'] = str(id)
    return example


def add_question(example, template):
    answer_list = example['answer_set']
    format_answer_list = []
    for answer in answer_list:
        if re.search(r'\d', answer['text']):
            format_answer_list.append(answer['text'])

    global id
    id = id + 1
    example['question'] = "How much {}?".format(example['path'])
    example['context'] = example['text']
    # example['answers'] = {
    #     'text': [example['value']],
    #     'answer_start': [example['text'].find(example['value'])]
    # }
    example['id'] = str(id)
    return example


def preprocess_validation_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    print('preprocess_validation_examples is :{}'.format(max_length))
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



# input = input.replace('\n',' ')

# input = '''
# The Group is principally engaged in education services. No substantial emissions are produced by
# combustion of any fuels in daily operation as the Group is not engaged in any industrial production.
# During the Reporting Period, the principal type of emission of the Group is exhaust generated by the
# Group’s self-owned vehicles. The main emission data are as follows:
# Major emissions Unit
# Emission
# volume
# Nitrogen oxide (NOx) Gram 673,012.0
# Sulphur dioxide (SOx) Gram 637.2
# Particulate Matter Gram 66,032.1
# '''

def get_question_set(example, template):
    answer_set = example['answer_set'][0]
    id = 0
    format_answer_set = {
        'question':[],
        'context':[],
        'id':[]
    }
    # print(answer_set)
    for answer in answer_set:
        # print(answer)
        id = id + 1
        format_answer_set['id'].append(str(id))
        format_answer_set['question'].append(template.format(answer['answer']))
        # print(answer['context'].split('[SEP]'))
        format_answer_set['context'].append(answer['context'].split('[SEP]')[1])

    return Dataset.from_dict(format_answer_set)

def question(eval_firuge_set, template):
    next_question_set = get_question_set(eval_firuge_set, template)

    eval_set = next_question_set.map(
        preprocess_validation_examples,
        batched=True,
        remove_columns=next_question_set.column_names,
    )
    print((len(next_question_set),len(eval_set)))

    result_set = extractor.get_fiture_set(next_question_set, eval_set, 5)
    return result_set

extractor = ESG(model_checkpoint)

test_data = load_dataset('csv', data_files = '/home/linzhisheng/esg/QA/report_all_eng.csv')
test_data = test_data['train']


# test_data = Dataset.from_dict(example)
small_eval_set = test_data.map(
    add_question_to_get_fiture, remove_columns=test_data.column_names)

eval_set = small_eval_set.map(
    preprocess_validation_examples,
    batched=True,
    remove_columns=small_eval_set.column_names,
    num_proc=32
)

# small_eval_set.save_to_disk('small_eval_set')
# eval_set.save_to_disk('eval_set')

# small_eval_set = load_from_disk('small_eval_set')
# eval_set = load_from_disk('eval_set')



print((len(small_eval_set),len(eval_set)))

eval_firuge_set = extractor.get_fiture_set(small_eval_set, eval_set, 1000)
eval_firuge_set = eval_firuge_set.remove_columns('context')
eval_firuge_set.to_csv('figure_1000_prob_all_128_32.csv',index=False)

# max_length = max_length + 50
# extractor.update_model('bert-large-uncased-whole-word-masking-finetuned-squad')

# result_set = question(eval_firuge_set,'What is the data {} about?')
# result_set.to_csv('data/figure_about.csv',index=False)
# # next question What is the unit of?
# result_set = question(eval_firuge_set,'What is the unit of {}?')
# result_set.to_csv('data/figure_unit.csv',index=False)
# # next question What year?
# result_set = question(eval_firuge_set,'What year is {} about?')
# result_set.to_csv('data/figure_year.csv',index=False)
