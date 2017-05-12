
import nltk
gaztteer = __import__('gaztteer')



def main():
    doc = "Greenpeace came up with the new study that says the coal-fired power plant at Rampal upazila in Bagerhat will increase 24-hour average ambient levels of nitrogen dioxide in nearby localities which is up to 25 percent over the current national urban average and sulphur oxide levels up to 50 percent over the urban " \
          "average.\"Emission limits for sulphur dioxide, nitrogen oxides, dust and mercury, as specified in the tender documents, are five to ten times higher than the best regulatory practice and technical state-of-the-art emission levels,\" it added." \
     "The plant could emit high levels of mercury, a potent neurotoxin that damages children’s brains and nervous systems and it could be" \
          " sufficient to render fish unsafe to eat over an area of approximately 70 square kilometres around the power plant.Additionally," \
          " 10,000kg of mercury over the life of the plant could end up in either the coal ash pond, which is subject to flooding, said Lauri" \
          " Myllyvirta of Greenpeace.Read Also: Not merely a forest but lifeThis additional mercury will pose further risks to the aquatic food" \
          " chain of the Sundarbans and the Bay of Bengal, impacting millions of people, the study added.National Committee to Protect the" \
          " Sundarbans convener Sultana Kamal, its member secretary Dr Abdul Matin and Doctors for Health and Environment President Dr Nazmun " \
          "Nahar also spoke at the press conference.The proposed 1,320-megawatt plant, a joint partnership between India's state-owned National" \
          " Thermal Power Corporation and Bangladesh Power Development Board, is on an area of over 1,834 acres of land and situated 14 kilometres" \
          " north of the world's largest mangrove forest the Sundarbans, which is a UNESCO world heritage site.Green groups have been opposing the " \
          "power plant since the signing of a memorandum of understanding between Bangladesh and India in 2010.The government, however, has been " \
          "emphatic on implementing the project and moving forward to implement the project, saying it was using the best technologies available " \
          "to prevent the possible damages.This additional mercury will pose further risks to the aquatic food chain of the Sundarbans and the Bay " \
          "of Bengal, impacting millions of people, the study added."


    temp = process_document(doc)


    print(nltk.corpus.treebank.tagged_sents()[22])
    print("==========")

    test1 = temp[0]
    singleArray = []


    for t in temp:
        singleArray.extend(t)

    print (singleArray)


    temp_ne = nltk.ne_chunk(singleArray)
    print (temp_ne)

    loc = gaztteer.LocationChunker()
    t = loc.parse(singleArray)

    # a = "For/IN the/DT past/JJ few/JJ days/NNS ,/, there/EX has/VBZ been/VBN a/DT rather/RB surprising/JJ change/NN in/IN the/DT way/NN things/NNS go/VBP on/IN at/IN the/DT Agargaon/NNP passport/NN office/NN in/IN the/DT capital/NN"

    print(t)
    print("===")
    print(loc.sub_leaves(t,'LOCATION'))



def process_document(document):
    s = nltk.sent_tokenize(document)
    s = [nltk.word_tokenize(a) for a in s]
    s = [nltk.pos_tag(a) for a in s]

    return s





if __name__=="__main__":
    main()

