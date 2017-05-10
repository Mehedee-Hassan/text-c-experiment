# from nltk.corpus.reader import WordListCorpusReader
# import enchant

# # reader = WordListCorpusReader('.', ['wordlist'])

# # reader.words()

# import nltk.data

# nltk.data.load('D:\\programming\\python\\text_classification_experiment\\book\\python3 text processing nltk3\\text_file\\myword.txt', format='raw')



# # d = enchant.Dict('en_US')
# # print(d.check("book"))


from nltk.corpus.reader import WordListCorpusReader
reader = WordListCorpusReader('C:/users/mhr/nltk_data',['myword.txt'])
print(reader.words())

print(reader.fileids())
