
import nltk
gaztteer = __import__('gaztteer')



def main():
    doc4 = "2010-09-07 00:00:00+06:00 8 transformer thieves get 14 yrs' RI each in Sylhet Sylhet Divisional Special Tribunal Sunday afternoon sentenced eight people to 14 years' RI each for lifting electric transformers. The convicts are Ripon Miah and Javed Miah of Munshipara and Gias Uddin, Liton Miah, Shukur Ali, Md. Khokan, Sohel Ahmed and Abdul Jalil of Bhatalia area of Sylhet city. Convicts Jalil and Sohel are at large while others, who were on bail appeared before the court on Sunday. According to the prosecution in brief, the people of village Bolaura of Sylhet Sadar upazila caught Ripon Miah red-handed while lifting transformer from an electric pole on 25 September of 2007. However, the others of the organised gang managed to escape. On quizzing Ripon disclosed all about the gang and their activities. He was later handed over to the police on filing a case the same night. Divisional Special Judge of Sylhet Hossain Helal passed the judgment. He also fined each of them Tk 25,000, in default, to suffer one year more in jail."
    doc5 = "Criminals exploded a crude bomb in front of Moynamoti Highway Police Station on Dhaka-Chittagong highway last night"
    doc2 = "Greenpeace came up with the new study that says the coal-fired power plant at Rampal upazila in Bagerhat will increase 24-hour average ambient levels of nitrogen dioxide in nearby localities which is up to 25 percent over the current national urban average and sulphur oxide levels up to 50 percent over the urban " \
          "average.\"Emission limits for sulphur dioxide, nitrogen oxides, dust and mercury, as specified in the tender documents, are five to ten times higher than the best regulatory practice and technical state-of-the-art emission levels,\" it added." \
     "The plant could emit high levels of mercury, a potent neurotoxin that damages children’s brains and nervous systems and it could be" \
          " sufficient to render fish unsafe to eat over an area of approximately 70 square kilometres around the power plant.Additionally," \
          " 10,000kg of mercury over the life of the plant could end up in either the coal ash pond, which is subject to flooding, said Lauri" \
          " Myllyvirta of Greenpeace.Read Also: Not merely a forest but lifeThis additional mercury will pose further risks to the aquatic food" \
          " chain of the Sundarban and the Bay of Bengal, impacting millions of people, the study added.National Committee to Protect the" \
          " Sundarban convener Sultana Kamal, its member secretary Dr Abdul Matin and Doctors for Health and Environment President Dr Nazmun " \
          "Nahar also spoke at the press conference.The proposed 1,320-megawatt plant, a joint partnership between India's state-owned National" \
          " Thermal Power Corporation and Bangladesh Power Development Board, is on an area of over 1,834 acres of land and situated 14 kilometres" \
          " north of the world's largest mangrove forest the Sundarban, which is a UNESCO world heritage site.Green groups have been opposing the " \
          "power plant since the signing of a memorandum of understanding between Bangladesh and India in 2010.The government, however, has been " \
          "emphatic on implementing the project and moving forward to implement the project, saying it was using the best technologies available " \
          "to prevent the possible damages.This additional mercury will pose further risks to the aquatic food chain of the Sundarban and the Bay " \
          "of Bengal, impacting millions of people, the study added."

    doc ="I was amazed to notice no protest from the readers; despite numerous news reports on housewives being killed by husbands & in-laws, due to non-receipt of committed dowries. Thank you Md. Tanjil, from the dept. of Geography and Environmental Studies, University of Chittagong, for writing against this social evil (source: The Daily Star, 03-05-2010). It usually happens among very poor citizens, living in remote villages. There is nobody to listen to them."
    doc2 = "2015-04-09 19:26:00+06:00 India\’s outsourcing giant guilty of fraud An Indian court on Thursday sentenced the" \
          " founder of an outsourcing giant and nine others to seven years in prison each after convicting them of stealing millions " \
          "of dollars in one of the largest frauds in the country's corporate history.Judge B.V.L.N. Chakravarthy also found Rama Raju," \
          " his two brothers and seven other officials of Satyam Computer Services guilty of cheating, using forged documents, falsifying " \
          "accounts and breach of trust.Federal investigators said the fraud by Raju and the others cost the company's shareholders 140 billion" \
          " rupees ($2.28 billion).Raju and other suspects were arrested in 2009 but released on bail while they awaited trial. Police took all the " \
          "10 convicts into custody after the verdict Thursday and sent them to a local prison.The judge also fined Raju 50 million rupees ($806,000)." \
          "The sentencing was planned for Friday, but judge Chakravarthy announced it Thursday at the request of the prosecutors.Satyam, which means \"truth\" " \
          "in Sanskrit was once India's fourth-largest software services company, counting a third of Fortune 500 companies and the U.S. government among its clients." \
          " It plunged into turmoil after Raju confessed in 2009 that he vastly inflated the company's assets by exaggerating cash balances, booking fake interest," \
          " and misstating both debt and liabilities.Tech Mahindra, a unit of the Mahindra Group, bought a majority stake in the company for $351 million and changed " \
          "its name to Mahindra Satyam in 2010."



    temp = process_document(doc)


    # print(nltk.corpus.treebank.tagged_sents()[22])
    print("==========")
    #
    # test1 = temp[0]
    singleArray = []


    for t in temp:
        singleArray.extend(t)

    print (singleArray)


    # temp_ne = nltk.ne_chunk(singleArray)
    # print (temp_ne)

    loc = gaztteer.LocationChunker()
    t = loc.parse(singleArray)

    # a = "For/IN the/DT past/JJ few/JJ days/NNS ,/, there/EX has/VBZ been/VBN a/DT rather/RB surprising/JJ change/NN in/IN the/DT way/NN things/NNS go/VBP on/IN at/IN the/DT Agargaon/NNP passport/NN office/NN in/IN the/DT capital/NN"

    print("===")
    print (t)
    print(loc.sub_leaves(t,'LOCATION'))




def sub_leaves(tree, label):
    return [t.leaves() for t in tree.subtrees(lambda s: s.label() == label)]



def process_document(document):
    s = nltk.sent_tokenize(document)
    s = [nltk.word_tokenize(a) for a in s]
    s = [nltk.pos_tag(a) for a in s]

    return s





if __name__=="__main__":
    main()





