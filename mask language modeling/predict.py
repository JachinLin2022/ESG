from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
from transformers import pipeline
from keybert import KeyBERT
from flair.embeddings import TransformerDocumentEmbeddings
import torch

model_checkpoint = "roberta-large"
local = './roberta-large-finetuned-esg-1w'

model = AutoModelForMaskedLM.from_pretrained(local)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

mask_filler = pipeline(
    "fill-mask", model=model,tokenizer = tokenizer
)
TEMPLATES = [
    'The sentence is about <mask>. ',
    'The Keyword is <mask>. ',
    'The keyword is <mask>. ',
    'The keyphrase is <mask>. ',
    'The key word is <mask>. ',
]

input = "As part of a strategic drive to reduce carbon emissions, Qantas are investigated collaborative partnerships with a number of leading renewable energy companies."

for T in TEMPLATES:
    mask_input = T + input
    
    inputs = tokenizer(mask_input, return_tensors="pt")
    token_logits = model(**inputs).logits
    # Find the location of [MASK] and extract its logits
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    mask_token_logits = token_logits[0, mask_token_index, :]
    # Pick the [MASK] candidates with the highest logits
    top_5_tokens = torch.topk(mask_token_logits, 50, dim=1).indices[0].tolist()

    # for token in top_5_tokens:
    print(f"{T}, keyword is [{tokenizer.decode(top_5_tokens)}]")
    # res = tokenizer(input)
    # output = model(**res)
    # print(output)


    # preds = mask_filler(mask_input)
    # for pred in preds:
    #     print(f">>> {pred['token_str']} >>> score is {pred['score']}")



# using key bert
hf_model = pipeline("feature-extraction", model=local, tokenizer = tokenizer)
kw_model = KeyBERT(model = hf_model)

keywords = kw_model.extract_keywords(input, top_n = 100)
print(keywords)
print(len(keywords))
