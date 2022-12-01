# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
from transformers import pipeline
from keybert import KeyBERT
from flair.embeddings import TransformerDocumentEmbeddings
import pandas as pd
import torch
import numpy as np


class ZeroShotClassifier:

    def create_zsl_model(self, model_name):
        """ Create the zero-shot learning model. """
        self.model = pipeline("zero-shot-classification",
                              model=model_name, device=0)

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

    # a smaller version: "microsoft/deberta-base-mnli"


# tokenizer.save_pretrained('roberta-large')
# origin_model = AutoModelForMaskedLM.from_pretrained('roberta-large')
# random_mask_model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
# torch.save(random_mask_model.state_dict(), 'lzs_test_save')
# origin_model.load_state_dict(torch.load('lzs_test_save'))
# random_mask_model.save
# origin_model.save_pretrained('./roberta-large2')

# model_name = "microsoft/deberta-v2-xlarge-mnli"
# candidate_labels = {'environment':'e', 'social':'s', 'company':'g'}
# ZSC = ZeroShotClassifier()
# ZSC.create_zsl_model(model_name)
# classified = ZSC.text_labels(input, candidate_labels)
# print(f'baseline: {classified}\n')
# input_list = input.replace(',','').replace('.','').split(' ')
# input_list = [s.lower() for s in input_list]
def predict(model, tokenizer, source):
    print('-----------------------------------')
    TEMPLATES = [
        # 'The Keyword is <mask>.',
        # 'The keyword is <mask>.',
        'In summary, the related word is <mask>.',
        # 'In summary, the keyword is <mask>.',
        # '<mask> is the keyword.',
        'In summary, <mask> is the keyword.',
        # 'In summary, <mask> is the related word.'
    ]
    average_radio = 0
    for input in source:
        input_list = input.replace(',', '').replace('.', '').split(' ')
        input_list = [s.lower() for s in input_list]
        # print(input_list)

        res_list = []
        for T in TEMPLATES:
            mask_input = T + input
            inputs = tokenizer(
                mask_input, return_tensors="pt", truncation=True)
            # print(inputs)
            token_logits = model(**inputs).logits
            # Find the location of [MASK] and extract its logits
            mask_token_index = torch.where(
                inputs["input_ids"] == tokenizer.mask_token_id)[1]
            # print(token_logits.shape)
            mask_token_logits = token_logits[0, mask_token_index, :]
            # Pick the [MASK] candidates with the highest logits
            top_tokens = torch.topk(
                mask_token_logits, 100, dim=1).indices[0].tolist()
            # print(top_tokens)
            top_tokens_list = tokenizer.decode(top_tokens).split(' ')
            top_tokens_list.pop(0)
            top_tokens_list = [s.lower() for s in top_tokens_list]

            intersect = set(input_list) & set(top_tokens_list)
            average_radio = average_radio + \
                len(intersect)/len(input_list)/len(source)
            # print(f"{T}:intersect:{intersect}, radio is {len(intersect)/len(input_list)}\n")
            print(f"{T}:{top_tokens_list}\n")

            # if len(res_list) == 0:
            #     res_list = top_tokens_list
            # else:
            #     tmp = []
            #     for word in res_list:
            #         for s in top_tokens_list:
            #             if word == s:
            #                 tmp.append(word)
            #                 break
            #     res_list = tmp
            # join_word = ''.join(x + ' ' for x in top_tokens_list)
            # classified = ZSC.text_labels(join_word, candidate_labels)
            # print(f'template:{T} res:{classified}\n')
            # for token in top_5_tokens:
            # print(f"{T}, keyword is [{tokenizer.decode(top_tokens)}]")
            # res = tokenizer(input)
            # output = model(**res)
            # print(output)

            # preds = mask_filler(mask_input)
            # for pred in preds:
            #     print(f">>> {pred['token_str']} >>> score is {pred['score']}")
    print(f'average:{average_radio}')
    # print(res_list)
    # using key bert
    # hf_model = pipeline("feature-extraction", model=model, tokenizer = tokenizer)
    # kw_model = KeyBERT(model = hf_model)

    # keywords = kw_model.extract_keywords(input, top_n = 100)
    # print(keywords)
    # print(len(keywords))


def test_predict():
    model_checkpoint = "esg-100w-model"
    local = './200w_mask_80'
    esg_tokenizer = AutoTokenizer.from_pretrained('roberta-esg-tokenizer')
    ori_tokenizer = AutoTokenizer.from_pretrained('roberta-large')
    source = pd.read_csv('source_all_english', nrows=100)
    input = ["Interpublics Directors are elected each year by Interpublics stockholders at the annual meeting of stockholders. Interpublics Corporate Governance Committee recommends nominees to the Board of Directors, and the Board proposes a slate of nominees to the stockholders for election."]
    input.append('To contribute to climate change mitigation, we actively explore opportunities to support local renewable energy generation. Solar panels are installed at Hang Seng 113 to generate renewable energy.')
    # input = source['Abstract'].tolist()
    # model = AutoModelForMaskedLM.from_pretrained(local)
    origin_model = AutoModelForMaskedLM.from_pretrained('roberta-large')
    random_mask_model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    predict(origin_model, ori_tokenizer, input)
    predict(random_mask_model, esg_tokenizer, input)
    # predict(model)


def label_data_result():
    model = AutoModelForMaskedLM.from_pretrained(local)
    device = torch.device('cuda:2')
    model = model.to(device)
    source = pd.read_csv('lable_data.csv')
    # for idx,row in source.iterrows():
    source['Abstract'] = 'In summary, the related word is <mask>. ' + \
        source['Abstract']
    source['vocab'] = source['Abstract']
    idx = 0
    # print(source['Abstract'].tolist())
    # inputs = tokenizer(source['Abstract'].tolist(), return_tensors="pt", truncation=True, padding=True)
    # token_logits = model(**inputs).logits
    # mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    # print(torch.where(inputs["input_ids"] == tokenizer.mask_token_id))
    # print(mask_token_index)
    # print(token_logits.shape)
    # exit(0)
    for text in source['Abstract'].tolist():
        # text = 'In summary, the related word is <mask>.' + row['Abstract']
        inputs = tokenizer(text, return_tensors="pt",
                           truncation=True, truncation_side='right').to(device)
        token_logits = model(**inputs).logits
        # Find the location of [MASK] and extract its logits
        mask_token_index = torch.where(
            inputs["input_ids"] == tokenizer.mask_token_id)[1]
        mask_token_logits = token_logits[0, mask_token_index, :]
        # Pick the [MASK] candidates with the highest logits
        top_tokens = torch.topk(mask_token_logits, 100,
                                dim=1).indices[0].tolist()
        top_tokens_list = tokenizer.decode(top_tokens).split(' ')
        top_tokens_list.pop(0)
        source['vocab'][idx] = '[' + \
            ''.join(x + ' ' for x in top_tokens_list) + ']'
        idx = idx + 1
        if idx % 10000 == 0:
            source.to_csv('lable_vocab.csv', index=False)
            print(idx)
        # print(f"{top_tokens_list}\n")
    source.to_csv('lable_vocab.csv', index=False)
    # print(source['Abstract'].tolist())
    # inputs = tokenizer(source['Abstract'].tolist(), return_tensors="pt", truncation=True)
    # token_logits = model(**inputs).logits
    # print(token_logits)


def c_tf_idf(documents, m, ngram_range=(1, 1)):
    count = CountVectorizer(ngram_range=ngram_range,
                            stop_words="english").fit(documents)
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)

    return tf_idf, count


def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
    words = count.get_feature_names()
    labels = list(docs_per_topic.Topic)
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {label: [(words[j], tf_idf_transposed[i][j])
                           for j in indices[i]][::-1] for i, label in enumerate(labels)}
    return top_n_words


def extract_topic_sizes(df):
    topic_sizes = (df.groupby(['Topic'])
                     .Doc
                     .count()
                     .reset_index()
                     .rename({"Topic": "Topic", "Doc": "Size"}, axis='columns')
                     .sort_values("Size", ascending=False))
    return topic_sizes


def cluster_model(input, model, output, min_cluster_size):
    embeddings = model.encode(input, show_progress_bar=True)
    # model = AutoModelForMaskedLM.from_pretrained(local)
    # feature_pipeline = pipeline('feature-extraction', model=model, tokenizer = tokenizer)
    # embeddings = feature_pipeline(['good', 'bad'])
    # model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    import umap
    umap_embeddings = umap.UMAP(n_neighbors=15,
                                n_components=5,
                                metric='cosine').fit_transform(embeddings)
    import hdbscan
    cluster = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                              metric='euclidean',
                              cluster_selection_method='eom').fit(umap_embeddings)

    import matplotlib.pyplot as plt

    # Prepare data
    umap_data = umap.UMAP(n_neighbors=15, n_components=2,
                          min_dist=0.0, metric='cosine').fit_transform(embeddings)
    result = pd.DataFrame(umap_data, columns=['x', 'y'])
    result['labels'] = cluster.labels_

    # Visualize clusters
    fig, ax = plt.subplots(figsize=(20, 10))
    outliers = result.loc[result.labels == -1, :]
    clustered = result.loc[result.labels != -1, :]
    plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=10)
    plt.scatter(clustered.x, clustered.y,
                c=clustered.labels, s=10, cmap='hsv_r')
    plt.colorbar()
    plt.savefig(output)

    docs_df = pd.DataFrame(input, columns=["Doc"])
    docs_df['Topic'] = cluster.labels_
    docs_df['Doc_ID'] = range(len(docs_df))
    docs_per_topic = docs_df.groupby(
        ['Topic'], as_index=False).agg({'Doc': ' '.join})

    tf_idf, count = c_tf_idf(docs_per_topic.Doc.values, m=len(input))
    top_n_words = extract_top_n_words_per_topic(
        tf_idf, count, docs_per_topic, n=5)
    topic_sizes = extract_topic_sizes(docs_df)
    print(topic_sizes.head(10))
    for t in topic_sizes.head(10)['Topic']:
        print(f"{t}: {top_n_words[t]}")

    # print(top_n_words)


def clustering():
    source = pd.read_csv('lable_vocab.csv', nrows=100)
    # print(source['vocab'])
    target = source['Abstract'].tolist()
    # print(target)
    # input = []
    # for vocab in target:
    #     # print(vocab[1:-2].split(' '))
    #     input = input + (vocab[1:-2].split(' '))
    #     # break

    # print(input)
    input = target

    from sentence_transformers import SentenceTransformer
    from transformers.pipelines import pipeline

    model1 = SentenceTransformer('roberta-large2')
    model2 = SentenceTransformer('roberta-large-finetuned-esg-100w')
    model3 = SentenceTransformer('200w_mask_80')
    for i in [5, 10, 15]:
        cluster_model(input, model1, str(i) + 'cluster/origin_text2.png', i)

        cluster_model(input, model2, str(i) + 'cluster/random_text2.png', i)

        cluster_model(input, model3, str(i) + 'cluster/dynamic_text2.png', i)
    # print(topic_sizes)

    # print(umap_embeddings)
# label_data_result()


test_predict()

# clustering()
