from nltk.tag import StanfordNERTagger
import os
import pandas as pd
# set java path in environment variables

java_path = r'G:\ProgramFiles\Java\jdk1.8.0_111\bin\java.exe'
os.environ['JAVAHOME'] = java_path
# load stanford NER

sn = StanfordNERTagger('D:/programming/python/stanford-ner-2016-10-31/classifiers/english.all.3class.distsim.crf.ser.gz',
path_to_jar='D:/programming/python/stanford-ner-2016-10-31/stanford-ner.jar')

# tag sentences
ne_annotated_sentences = [sn.tag(sent) for sent in tokenized_sentences]
# extract named entities
named_entities = []
for sentence in ne_annotated_sentences:
    temp_entity_name = ''
    temp_named_entity = None

    for term, tag in sentence:
        # get terms with NE tags
        if tag != 'O':
            temp_entity_name = ' '.join([temp_entity_name, term]).strip()
            #            get NE name
            temp_named_entity = (temp_entity_name, tag)
            # get NE and itscategory

        else:
            if temp_named_entity:
                named_entities.append(temp_named_entity)
                temp_entity_name = ''
                temp_named_entity = None

# get unique named entities
named_entities = list(set(named_entities))
# store named entities in a data frame
entity_frame = pd.DataFrame(named_entities,
columns=['Entity Name', 'Entity Type'])