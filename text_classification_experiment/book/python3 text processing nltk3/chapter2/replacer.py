from nltk.chunk import ChunkParserI
from nltk.chunk.util import conlltags2tree
from nltk.corpus import names
from nltk.chunk import chunk_ne_sents

class PersonChunker(ChunkParserI):
    def __init__(self):
        self.name_set = set(names.words())


    def parse(self, tagged_sent):
        iobs = []
        in_person = False
        
        for word, tag in tagged_sent:
            if word in self.name_set and in_person:
                iobs.append((word, tag, 'I-PERSON'))
            
            elif word in self.name_set:
                iobs.append((word, tag, 'B-PERSON'))
                in_person = True
            else:
                iobs.append((word, tag, 'O'))
                in_person = False
        return conlltags2tree(iobs)

# chunker = PersonChunker()
# sub_leaves(chunker.parse(treebank_chunk.tagged_sents()[0]),
# 'PERSON')

ne_chunk(treebank_chunk.tagged_sents()[0], binary=True)

