from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
from transformers import pipeline
from keybert import KeyBERT
from flair.embeddings import TransformerDocumentEmbeddings
import torch

class ZeroShotClassifier:

    def create_zsl_model(self, model_name):
        """ Create the zero-shot learning model. """
        self.model = pipeline("zero-shot-classification", model=model_name, device = 0)
    
        
    def classify_text(self, text, categories):
        """
        Classify text(s) to the pre-defined categories using a
        zero-shot classification model and return the raw results.
        """
        # Classify text using the zero-shot transformers model
        hypothesis_template = "This text is about {}."
        result = self.model(text, categories, multi_label=True,
                            hypothesis_template=hypothesis_template)
        return result

    
    def text_labels(self, text, category_dict, cutoff=None):
        """
        Classify a text into the pre-defined categories. If cutoff
        is defined, return only those entries where the score > cutoff
        """
        # Run the model on our categories
        categories = list(category_dict.keys())
        
        result = (self.classify_text(text, categories))
        return result
        # Format as a pandas dataframe and add ESG label
        # df = pd.DataFrame(result).explode(['labels', 'scores'])
        # df["ESG"] = df.labels.map(category_dict)
        # # If a cutoff is provided, filter the dataframe
        # if cutoff:
        #     df = df[df.scores.gt(cutoff)].copy()
        # return df.reset_index(drop=True)

# Define and Create the zero-shot learning model
model_name = "microsoft/deberta-v2-xlarge-mnli" 
    # a smaller version: "microsoft/deberta-base-mnli"
ZSC = ZeroShotClassifier()
ZSC.create_zsl_model(model_name)
candidate_labels = {'environment':'e', 'social':'s', 'company':'g'}

model_checkpoint = "roberta-large-finetuned-esg-100w"
local = './200w_mask'

origin_model = AutoModelForMaskedLM.from_pretrained('roberta-large')
random_mask_model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
# torch.save(random_mask_model.state_dict(), 'lzs_test_save')
# origin_model.load_state_dict(torch.load('lzs_test_save'))
# random_mask_model.save
model = AutoModelForMaskedLM.from_pretrained(local)
tokenizer = AutoTokenizer.from_pretrained('roberta-large')

input = "Interpublics Directors are elected each year by Interpublics stockholders at the annual meeting of stockholders. Interpublics Corporate Governance Committee recommends nominees to the Board of Directors, and the Board proposes a slate of nominees to the stockholders for election."
classified = ZSC.text_labels(input, candidate_labels)
print(f'baseline: {classified}\n')

def predict(model):
    mask_filler = pipeline(
        "fill-mask", model=model,tokenizer = tokenizer
    )
    TEMPLATES = [
        # ' The Keyword is <mask>. ',
        # ' The keyword is <mask>.',
        ' In summary, the related word is <mask>.',
        ' In summary, the keyword is <mask>.',
        ' <mask> is the keyword.',
    ]
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
        # print(f"{T}:{top_tokens_list}\n")



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

        join_word = ''.join(x + ' ' for x in top_tokens_list)
        classified = ZSC.text_labels(join_word, candidate_labels)
        print(f'template:{T} res:{classified}\n')
        # for token in top_5_tokens:
        # print(f"{T}, keyword is [{tokenizer.decode(top_tokens)}]")
        # res = tokenizer(input)
        # output = model(**res)
        # print(output)


        # preds = mask_filler(mask_input)
        # for pred in preds:
        #     print(f">>> {pred['token_str']} >>> score is {pred['score']}")

    # print(res_list) 
    # # using key bert
    # hf_model = pipeline("feature-extraction", model=model, tokenizer = tokenizer)
    # kw_model = KeyBERT(model = hf_model)

    # keywords = kw_model.extract_keywords(input, top_n = 100)
    # print(keywords)
    # print(len(keywords))



predict(origin_model)
predict(random_mask_model)
predict(model)





