from transformers import pipeline
import torch
# Replace this with your own checkpoint
# model_checkpoint = "deepset/roberta-base-squad2"
model_checkpoint = "bert-large-uncased-whole-word-masking-finetuned-squad"
question_answerer = pipeline("question-answering", model='bert-finetuned-esgQA-fine-grained-8-2')


question_answerer2 = pipeline("question-answering", model=model_checkpoint)
context = """

The Group is principally engaged in education services. No substantial emissions are produced by
combustion of any fuels in daily operation as the Group is not engaged in any industrial production.
During the Reporting Period, the principal type of emission of the Group is exhaust generated by the
Group’s self-owned vehicles. The main emission data are as follows:
Major emissions Unit
Emission
volume
Nitrogen oxide (NOx) Gram 673, 012. 0
Sulphur dioxide (SOx) Gram 637. 2
Particulate Matter Gram 66, 032. 1

"""
question = "How much emission?"

t = question_answerer(question=question, context=context)
print(t)
    # print(question_answerer(question=question, context=context))

# question = "How much ResourceUse?"
# print(question_answerer(question=question, context=context))

# answer = question_answerer(question=question, context=context)['answer']

question = "What is the unit of {}?".format('673,012')
print(question_answerer2(question=question, context=context))

question = "What is the data {} about?".format('673,012')
print(question_answerer2(question=question, context=context))