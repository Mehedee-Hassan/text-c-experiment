import nltk.data

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
fp = open("D:/google_drive/MSc Research/implement/nlp python/classification/classification1/svm_test/test_classification1/data/crime/1.txt")
data = fp.read()
print ('\n-----\n'.join(tokenizer.tokenize(data)))