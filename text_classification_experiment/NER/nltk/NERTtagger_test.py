

from nltk.tag.stanford import NERTagger


st = NERTagger('/usr/share/stanford-ner/classifiers/all.3class.distsim.crf.ser.gz',
               '/usr/share/stanford-ner/stanford-ner.jar')
st.tag('Rami Eid is studying at Stony Brook University in NY'.split())