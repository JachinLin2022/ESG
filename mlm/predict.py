from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
from transformers import pipeline
from keybert import KeyBERT
from flair.embeddings import TransformerDocumentEmbeddings
import torch

model_checkpoint = "roberta-large"
local = './roberta-large-finetuned-esg-100w'

origin_model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
model = AutoModelForMaskedLM.from_pretrained(local)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def predict(model):
    mask_filler = pipeline(
        "fill-mask", model=model,tokenizer = tokenizer
    )
    TEMPLATES = [
        # 'The sentence is about <mask>. ',
        # 'The Keyword is <mask>. ',
        ' The keyword is <mask>.',
        # 'The keyphrase is <mask>. ',
        # 'The key word is <mask>. ',
        # ' In summary, the sentence is about <mask>.',
        ' In summary, the related word is <mask>.',
        ' In summary, the keyword word is <mask>.'
    ]

    input = "These individuals conduct safety inspections to eliminate hazards and provide first-aid assistance in the event of an accident. "
    res_list = []
    for T in TEMPLATES:
        mask_input =  input + T
        
        inputs = tokenizer(mask_input, return_tensors="pt")
        token_logits = model(**inputs).logits
        # Find the location of [MASK] and extract its logits
        mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
        mask_token_logits = token_logits[0, mask_token_index, :]
        # Pick the [MASK] candidates with the highest logits
        top_tokens = torch.topk(mask_token_logits, 50, dim=1).indices[0].tolist()
        top_tokens_list = tokenizer.decode(top_tokens).split(' ')
        top_tokens_list.pop(0)
        print(f"{T}:{top_tokens_list}\n")



        if len(res_list) == 0:
            res_list = top_tokens_list
        else:
            tmp = []
            for word in res_list:
                for s in top_tokens_list:
                    if word == s:
                        tmp.append(word)
                        break
            res_list = tmp


        # for token in top_5_tokens:
        # print(f"{T}, keyword is [{tokenizer.decode(top_tokens)}]")
        # res = tokenizer(input)
        # output = model(**res)
        # print(output)


        # preds = mask_filler(mask_input)
        # for pred in preds:
        #     print(f">>> {pred['token_str']} >>> score is {pred['score']}")

    print(res_list)
    # using key bert
    hf_model = pipeline("feature-extraction", model=model, tokenizer = tokenizer)
    kw_model = KeyBERT(model = hf_model)

    keywords = kw_model.extract_keywords(input, top_n = 100)
    print(keywords)
    print(len(keywords))
predict(origin_model)
predict(model)

