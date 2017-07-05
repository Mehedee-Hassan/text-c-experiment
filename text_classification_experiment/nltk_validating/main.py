import sys, os


# getting the real path of dir
import nltk

projectpath = os.path.dirname(os.path.realpath('main.py'))

print ("project path = ",projectpath)

libpath = projectpath + '/lib_cosine'
writeresultpath = projectpath + '/writefiles'
sentence_lib = projectpath + '/sentence_with_loc'

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')



sys.path.append(libpath)
sys.path.append(sentence_lib)
sys.path.append(projectpath+"/lib_cosine")

os.chdir(projectpath)

import CommonNames as CN

import ne_filter  as SWL


from pprint import  pprint
from pymongo import MongoClient

import temp_query_cosine as qur
from bson.objectid import ObjectId
import k_means
import hierarchical_clustering as hierarchical

import nltk
import gazetteer as gaztteer


# database collection settings
import CommonNames as CN

client = MongoClient()
db = CN.getDatabase(client)
index_col_name = CN.indexCollectionName()
document_col_name = CN.documentCollectionName()
tfvectDoc_col_name = CN.tfvCollectionName()


# ==============================




indexCollection = db[index_col_name]
documentCollection = db[document_col_name]

tf_docVector = db[tfvectDoc_col_name]




all_sentence_file =  writeresultpath +'\main_file'
location_sentence_file =  writeresultpath +'\location_file'



def readAndWrite():


    docnum = -1
    line = ''


    sum1 = 0
    for data in db[document_col_name].find().limit(100):
        docnum += 1
        text = data ['data']

        sents = tokenizer.tokenize(text)

        reader1 = open(all_sentence_file, "a", encoding="utf8")
        reader2 = open(location_sentence_file, "a", encoding="utf8")

        for s in sents:
            str2 = str(docnum) + '|' + s +'\n\n'
            # reader1.write(str2)

            temp ,list = preTrainedNLTK(s)

            sum1 += temp

            print("sum = ",sum1)

            if len(list)!=0:
                print(docnum)
                # reader2.write("nltk "+str(docnum)+"|"+s+"\n\n")


        # sents = SWL.line_token(text,docnum)
        #
        # for s in sents:
        #     reader2.write("nltk "+str(docnum)+"|"+s+"\n\n")


        reader1.close()
        reader2.close()





#  text = nltk.word_tokenize("my name is Mehedee and i live in Dhaka")
# >>> test = nltk.pos_tag(text)
# >>> aa = nltk.ne_chunk(test)
# >>> aa

def preTrainedNLTK(sentence):

    s1 = nltk.word_tokenize(sentence)
    # print(s1)
    print("len =",len(s1))
    words = nltk.pos_tag(s1)
    tree =  nltk.ne_chunk(words)

    loc = []
    for a in tree.subtrees():
        if a.label() == 'GPE' or a.label() == 'LOCATION':
            print(a[0][0])
            loc.append(a[0][0])

    return len(s1),loc



def gazett(doc):



    temp = process_document(doc)


    # print(nltk.corpus.treebank.tagged_sents()[22])
    print("==========")
    #
    # test1 = temp[0]
    singleArray = []


    for t in temp:
        singleArray.extend(t)

    # print (singleArray)


    # temp_ne = nltk.ne_chunk(singleArray)
    # print (temp_ne)

    loc = gaztteer.LocationChunker()
    t = loc.parse(singleArray)

    # a = "For/IN the/DT past/JJ few/JJ days/NNS ,/, there/EX has/VBZ been/VBN a/DT rather/RB surprising/JJ change/NN in/IN the/DT way/NN things/NNS go/VBP on/IN at/IN the/DT Agargaon/NNP passport/NN office/NN in/IN the/DT capital/NN"

    # print("===")
    # print (t)

    location_name = loc.sub_leaves(t,'LOCATION')
    gpe_name = loc.sub_leaves(t,'GPE')



    for lname in location_name:
        print(lname[0][0])


    for lname in gpe_name:
        print(lname[0][0])


    return  loc.sub_leaves(t,'LOCATION')



def sub_leaves(tree, label):
    return [t.leaves() for t in tree.subtrees(lambda s: s.label() == label)]



def process_document(document):
    s = nltk.sent_tokenize(document)
    s = [nltk.word_tokenize(a) for a in s]
    s = [nltk.pos_tag(a) for a in s]

    return s


if __name__ == '__main__':
    readAndWrite()