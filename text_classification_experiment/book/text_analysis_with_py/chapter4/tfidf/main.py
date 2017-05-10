from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import numpy as np
import pandas as pd 



CORPUS = [
'the sky is blue',
'sky is blue and sky is beautiful',
'the beautiful sky is so blue',
'i love blue cheese'
]

new_doc = ['loving this blue sky today']




def tfidf_transformer(bow_matrix):
    transformer = TfidfTransformer(norm='l2',smooth_idf=True,use_idf=True)
    tfidf_matrix = transformer.fit_transform(bow_matrix)
    return transformer, tfidf_matrix




def bow_extractor(corpus,ngram_range=(1,1)):
    v = CountVectorizer(min_df=1,ngram_range=ngram_range)
    feature = v.fit_transform(corpus)

    return v,feature



def main():
    bow_vectorizer ,bow_f = bow_extractor(CORPUS)
    f = bow_f.todense()

    
    # print (f)
    # fnames = bow_vectorizer.get_feature_names()
    # display_features(f,fnames)
    
    
    # print("\n\n")


    new_doc_feature = bow_vectorizer.transform(new_doc)
    new_doc_feature = new_doc_feature.todense()



    # print("=====\nnew doc features")
    # print (new_doc_feature)

    # fnames = bow_vectorizer.get_feature_names()
    # display_features(new_doc_feature,fnames)

    # tf idf
    tf_idf(bow_vectorizer,bow_f,new_doc_feature)



def tf_idf(bow_vectorizer,bow_feature,new_doc_features):
    

    feature_names = bow_vectorizer.get_feature_names()
    tfidf_trans, tdidf_features = tfidf_transformer(bow_feature)

    feature = np.round(tdidf_features.todense(), 2)

    display_features(feature,feature_names)

    nd_tfidf = tfidf_trans.transform(new_doc_features)
    nd_features = np.round(nd_tfidf.todense(), 2)


    display_features(nd_features, feature_names)





def display_features(features,features_names):
    df = pd.DataFrame(data=features,columns=features_names)
    pd.set_option('display.max_columns', None)

    print (df)




#  using direct implementation of tf-idf
def tfidf_extractor(corpus, ngram_range=(1,1)):


    vectorizer = TfidfVectorizer(min_df=1,norm='l2',smooth_idf=True, use_idf=True, ngram_range=ngram_range)

    features = vectorizer.fit_transform(corpus)
    
    
    return vectorizer, features



if __name__=="__main__":
    main()
