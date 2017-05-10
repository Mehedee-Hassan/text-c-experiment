
import gensim
import nltk


CORPUS = [
'the sky is blue',
'sky is blue and sky is beautiful',
'the beautiful sky is so blue',
'i love blue cheese'
]

new_doc = ['loving this blue sky today']


def main():
    
    # tokenize corpora
    TOKENIZED_CORPUS = [nltk.word_tokenize(sentence) for sentence in CORPUS]
    tokenized_new_doc = [nltk.word_tokenize(sentence) for sentence in new_doc]
    # build the word2vec model on our training corpus
    model = gensim.models.Word2Vec(TOKENIZED_CORPUS, size=10, window=10,   min_count=2, sample=1e-3)




if __name__ == "__main__":
    main()