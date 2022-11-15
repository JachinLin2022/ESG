import nltk
# nltk.download('averaged_perceptron_tagger')
for i in range(10000):
    sent="To mitigate the effects of global warming, we have been using eco-friendly refrigerants in our new air-conditioning systems."
    tokens = nltk.word_tokenize(sent)
    print(tokens)
    taged_sent = nltk.pos_tag(tokens)
    print(taged_sent) 