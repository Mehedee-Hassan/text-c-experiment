
from nltk.corpus.reader import CategorizedPlaintextCorpusReader


def main():
    path= "/data/"

    file =open("data/movie_neg.txt",'r')
    fr = file.read()

    print (fr)


    reader = CategorizedPlaintextCorpusReader('.', r'data/movie_.*\.txt', cat_pattern=r'data/movie_(\w+)\.txt')
    
    print (reader.categories())



if __name__ == "__main__":
    main()