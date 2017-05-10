
from nltk.chunk import ChunkParserI
from nltk.chunk.util import conlltags2tree
from nltk.corpus import gazetteers


class LocationChunker(ChunkParserI):
    # '''Chunks locations based on the gazetteers corpus.
    # >>> loc = LocationChunker()
    # >>> t = loc.parse([('San', 'NNP'), ('Francisco', 'NNP'), ('CA', 'NNP'), ('is', 'BE'), ('cold', 'JJ'), ('compared', 'VBD'), ('to', 'TO'), ('San', 'NNP'), ('Jose', 'NNP'), ('CA', 'NNP')])
    # >>> sub_leaves(t, 'LOCATION')
    # [[('San', 'NNP'), ('Francisco', 'NNP'), ('CA', 'NNP')], [('San', 'NNP'), ('Jose', 'NNP'), ('CA', 'NNP')]]
    # '''

    def __init__(self):
        # gazetteers is a WordListCorpusReader of many different location words
        self.locations = set(gazetteers.words())

        self.locations.add('Chapai Nabab Ganj')
        print(self.locations)

        self.lookahead = 0
        # need to know how many words to lookahead in the tagged sentence to find a location
        for loc in self.locations:
            nwords = loc.count(' ')

            if nwords > self.lookahead:
                self.lookahead = nwords

    def iob_locations(self, tagged_sent):
        i = 0
        l = len(tagged_sent)
        inside = False

        while i < l:
            word, tag = tagged_sent[i]
            j = i + 1
            k = j + self.lookahead
            nextwords, nexttags = [], []
            loc = False
            # lookahead in the sentence to find multi-word locations
            while j < k:
                if ' '.join([word] + nextwords) in self.locations:
                    # combine multiple separate locations into single location chunk
                    if inside:
                        yield word, tag, 'I-LOCATION'
                    else:
                        yield word, tag, 'B-LOCATION'
                    # every next word is inside the location chunk
                    for nword, ntag in zip(nextwords, nexttags):
                        yield nword, ntag, 'I-LOCATION'
                    # found a location, so we're inside a chunk
                    loc, inside = True, True
                    # move forward to the next word since the current words
                    # are already chunked
                    i = j
                    break

                if j < l:
                    nextword, nexttag = tagged_sent[j]
                    nextwords.append(nextword)
                    nexttags.append(nexttag)
                    j += 1
                else:
                    break
            # if no location found, then we're outside the location chunk
            if not loc:
                inside = False
                i += 1
                yield word, tag, 'O'

    def parse(self, tagged_sent):
        iobs = self.iob_locations(tagged_sent)
        return conlltags2tree(iobs)


def sub_leaves(tree, label):
    return [t.leaves() for t in tree.subtrees(lambda s: s.label() == label)]



# run script
# ==========

loc = LocationChunker()
t =  loc.parse([('Chapai', 'NNP'), ('Nabab', 'NNP'), ('Ganj','NNP'), ('is', 'BE'), ('cold', 'JJ'), ('compared', 'VBD'), ('to','TO'), ('San', 'NNP'), ('Jose', 'NNP'), ('CA', 'NNP')])

print(t)
print("===")
print(sub_leaves(t,'LOCATION'))