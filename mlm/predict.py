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
import spacy
from wordcloud import WordCloud
nlp = spacy.load('en_core_web_sm')

def predict(model, tokenizer, source,topk, name):
    print(topk)
    if name.find('finbert') >= 0:
        mask_token  = '[MASK]'
    else:
        mask_token  = '<mask>'
    TEMPLATES = [
        f'{mask_token} is the keyphrase. ',
        f'{mask_token} is the keyword. ',
        f'In summary, {mask_token} is the keyphrase. ',
        f'In summary, {mask_token} is the keyword. '
    ]
    average_radio = 0
    result = {}
    first = 1
    for input in source:
        # input_list = input.replace(',', '').replace('.', '').split(' ')
        # input_list = [s.lower() for s in input_list]
        # print(input_list)
        # print(print(input))
        for T in TEMPLATES:
            mask_input = T + input
            inputs = tokenizer(
                mask_input, return_tensors="pt", truncation=True).to(torch.device('cuda:0'))
            # print(inputs)
            token_logits = model(**inputs).logits
            # Find the location of [MASK] and extract its logits
            mask_token_index = torch.where(
                inputs["input_ids"] == tokenizer.mask_token_id)[1]
            # print(token_logits.shape)
            mask_token_logits = token_logits[0, mask_token_index, :]
            # Pick the [MASK] candidates with the highest logits
            top_tokens = torch.topk(
                mask_token_logits, topk, dim=1).indices[0].tolist()
            # print(top_tokens)
            top_tokens_list = tokenizer.decode(top_tokens).split(' ')
            # top_tokens_list.pop(0)
            tmp = []
            for s in top_tokens_list:
                if s.lower() not in ['this','it','who','which','where','that','the','and','mwh','kwh']:
                    tmp.append(s.lower())
            top_tokens_list = tmp

            for token in top_tokens_list:
                if token in result.keys():
                    result[token] = result[token] + 1
                else:
                    result[token] = 1
            # intersect = set(input_list) & set(top_tokens_list)
            # average_radio = average_radio + \
            #     len(intersect)/len(input_list)/len(source)
            # print(f"{T}:intersect:{intersect}, radio is {len(intersect)/len(input_list)}\n")
            
            # print(f"{T}:{top_tokens_list}")
            
            # for doc in nlp.pipe(top_tokens_list):
            #     for token in doc:
            #         print(token.tag_)
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
        # print('\n\n==================================================')
    # print(f'average:{average_radio}')
    print(len(result))
    wordcloud = WordCloud(width=1600, height=1600,background_color="white",max_words = 10000).generate_from_frequencies(result)
    wordcloud.to_file(f'wordcloud/{name}{topk}.png')
    print(f'top-k is {topk}, res is: {sorted(result.items(), key = lambda x:x[1],reverse = True)[:1000]}')
    # using key bert
    # hf_model = pipeline("feature-extraction", model=model, tokenizer = tokenizer)
    # kw_model = KeyBERT(model = hf_model)

    # keywords = kw_model.extract_keywords(input, top_n = 100)
    # print(keywords)
    # print(len(keywords))


def test_predict():
    esg_tokenizer = AutoTokenizer.from_pretrained('roberta-esg-tokenizer')
    ori_tokenizer = AutoTokenizer.from_pretrained('roberta-large')
    source = pd.read_csv('source_all_english', nrows=10000)
    # source = source[40000:41000]
    input = []
    # input.append("Interpublics Directors are elected each year by Interpublics stockholders at the annual meeting of stockholders. Interpublics Corporate Governance Committee recommends nominees to the Board of Directors, and the Board proposes a slate of nominees to the stockholders for election.")
    input.append('To contribute to climate change mitigation, we actively explore opportunities to support local renewable energy generation. Solar panels are installed at Hang Seng 113 to generate renewable energy.')
    input = source['Abstract'].tolist()
    
    # for name in ['esg-roberta-dynamic_80_10_10-model','esg-roberta-random-model', 'esg-roberta-dynamic_80_ROOT-model']:
    #     model = AutoModelForMaskedLM.from_pretrained(name).to(torch.device('cuda:0'))
    #     # origin_model = AutoModelForMaskedLM.from_pretrained('roberta-large').to(torch.device('cuda:0'))
    #     # random_mask_model = AutoModelForMaskedLM.from_pretrained('esg-roberta-random-model').to(torch.device('cuda:0'))
    #     # dynamic_mask_model = AutoModelForMaskedLM.from_pretrained('esg-roberta-dynamic-model').to(torch.device('cuda:0'))
    #     # predict(origin_model, ori_tokenizer, input)
    #     # for i in range(3):
    #     #     predict(origin_model, ori_tokenizer, input, 100*pow(10,i))
    #     # print(input[0])
    #     # model = AutoModelForMaskedLM.from_pretrained('yiyanghkust/finbert-pretrain').to(torch.device('cuda:0'))
    #     # t = AutoTokenizer.from_pretrained('yiyanghkust/finbert-pretrain',model_max_length=512)
    #     for i in range(2):
    #         predict(model, esg_tokenizer, input, 100*pow(10,i), name)
            
    model = AutoModelForMaskedLM.from_pretrained('yiyanghkust/finbert-pretrain').to(torch.device('cuda:0'))
    t = AutoTokenizer.from_pretrained('yiyanghkust/finbert-pretrain',model_max_length=512)
    for i in range(2):
        predict(model, t, input, 100*pow(10,i), 'finbert')

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
