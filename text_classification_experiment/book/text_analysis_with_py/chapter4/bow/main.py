from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd 

CORPUS = [
'the sky is blue',
'sky is blue and sky is beautiful',
'the beautiful sky is so blue',
'i love blue cheese'
]

new_doc = ['loving this blue sky today']






def bow_extractor(corpus,ngram_range=(1,1)):
    v = CountVectorizer(min_df=1,ngram_range=ngram_range)
    feature = v.fit_transform(corpus)

    return v,feature



def main():
    bow_vectorizer ,bow_f = bow_extractor(CORPUS,(1,3))
    f = bow_f.todense()

    
    print (f)
    fnames = bow_vectorizer.get_feature_names()
    display_features(f,fnames)
    
    
    print("\n\n")


    new_doc_feature = bow_vectorizer.transform(new_doc)

    new_doc_feature = new_doc_feature.todense()

    print("=====\nnew doc features")
    print (new_doc_feature)

    fnames = bow_vectorizer.get_feature_names()
    display_features(new_doc_feature,fnames)




def display_features(features,features_names):
    df = pd.DataFrame(data=features,columns=features_names)
    pd.set_option('display.max_columns', None)

    print (df)

















if __name__=="__main__":
    main()
