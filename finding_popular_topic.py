import time
import pre_process
from gensim import corpora, models
import gensim


ALG = "LDA"
PRE_PROCESS_TYPE = 'remove_twitter_account'


def run_lda():
    processed_docs = pre_process.get_input(PRE_PROCESS_TYPE)
    dictionary = gensim.corpora.Dictionary(processed_docs)
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]

    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)

    for idx, topic in lda_model.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic))

    print(processed_docs[4310])
    for index, score in sorted(lda_model[bow_corpus[4310]], key=lambda tup: -1 * tup[1]):
        print("\nScore: {}\t \nTopic {}: {}".format(score, index, lda_model.print_topic(index, 10)))