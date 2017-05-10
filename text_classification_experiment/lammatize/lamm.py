from nltk.stem import WordNetLemmatizer


wordnet_lemmatizer = WordNetLemmatizer()

arr = ["maximum saying","presumably saying","provision cement"]

print (wordnet_lemmatizer.lemmatize("saying"))

a = [wordnet_lemmatizer.lemmatize(item) for item in arr]

print(a)