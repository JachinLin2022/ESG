from transformers import pipeline
import torch
# Replace this with your own checkpoint
# model_checkpoint = "deepset/roberta-base-squad2"
model_checkpoint = "bert-large-uncased-whole-word-masking-finetuned-squad"
question_answerer = pipeline("question-answering", model='bert-finetuned-esgQA-fine-grained-8-2', device=torch.device('cuda:1'))


# question_answerer2 = pipeline("question-answering", model=model_checkpoint)
context = """

"Water Management
[G4: 303-1  303-3]

Our conservation and consolidation projects reduced water consumption by 2.2 percent in 2016 compared to the annual consumption in 2015. In total,
we withdrew 11,218,076,415 liters of water and recycled or reused 5,446,796,235 liters (49 percent of total quantity withdrawn)."



"""
question = "How much FreshWaterWithdrawalTotal?"
for i in range(10000):
    t = question_answerer(question=question, context=context)
    # print(question_answerer(question=question, context=context))

# question = "How much ResourceUse?"
# print(question_answerer(question=question, context=context))

# answer = question_answerer(question=question, context=context)['answer']

# question = "What is the unit of {}?".format(answer)
# print(question_answerer2(question=question, context=context))

# question = "What is the data {} about?".format(answer)
# print(question_answerer2(question=question, context=context))