import nltk

nltk.download('movie_reviews')

from nltk.corpus import movie_reviews
from nltk.stem import PorterStemmer
import re
import math

stemmer = PorterStemmer()

#do the tf-idf calculations first to create a table of values
#then use the table for the cosine similarity

#sample to use for testing
doc_1 = ['food', 'restaurant', 'Customer', '3', 'restaurant', 'waitress', '.', "", "''"]
doc_2 = ['food', 'Store', '1000', 'customer', '!', 'cashier']

#to be used on the list of pos review words and list of neg review words
#aka 2 'documents'


def compare_similarities_2docs(document1, document2):

    #preparing and merging documents into one list:
    print('Prepping documents...', end = '\n\n')
    docs_mod = [[] for i in range(2)]

    #document 1
    for i in range(len(document1)):
        match = re.match('[A-Za-z]', document1[i])
        if match:
            docs_mod[0].append(document1[i].lower())

    #document 2
    for i in range(len(document2)):
        match = re.match('[A-Za-z]', document2[i])
        if match:
            docs_mod[1].append(document2[i].lower())


    #creates list of all terms in documents:
    print('Creating list of all unique terms...', end = '\n\n')
    terms = []

    for i in range(2):
        for j in range(len(docs_mod[i])):
            if docs_mod[i][j] not in terms:
                terms.append(docs_mod[i][j])


    #calculate tf values for words in both documents:
    print('Calculating TF_t values...')
    print('This may take awhile...please wait', end = '\n\n')
    tf = [[] for i in range(len(terms))]

    for i in range(len(tf)): #for tf_raw
        for k in range(2):
            count = 0
            for m in range(len(docs_mod[k])):
                if docs_mod[k][m] == terms[i]:
                    count = count + 1
            tf[i].append(math.log10(count+1))


    #calculate idf values for words in both documents:
    print('Calculating IDF_t values...')
    print('This may take awhile...please wait', end = '\n\n')
    #df = []
    idf = []

    #total number of documents
    N = 2

    for i in range(len(terms)):
        count = 0
        for j in range(2):
            if terms[i] in docs_mod[j]:
                count = count + 1
        idf.append(math.log10(N/count))

    #calculate tf-idf using previously calculate values from tf and idf:
    print('Calculating TF-IDF_t values...')
    print('This may take awhile...please wait', end = '\n\n')
    tf_idf = [[] for i in range(len(terms))]

    for i in range(len(tf_idf)):
        for j in range(2):
            tf_idf[i].append(tf[i][j]*idf[i])


    #print the tf-idf results in a table for all terms in documents:  NOT NEEDED for this
    data = [[] for i in range(len(terms))]

    for i in range(len(terms)):
        data[i].append(terms[i])
        for j in range(2):
            data[i].append(tf_idf[i][j])

    #calculating the cosine similarities:
    print('Calculating cosine similarities and average similarity...')
    print('This may take awhile...please wait', end = '\n\n')
    avg_sim = 0
    avg_count = 0

    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            cos_sim_calc = (data[i][1]*data[j][1]+data[i][2]*data[j][2])
            if cos_sim_calc != 0:
                cos_sim_calc = cos_sim_calc/(math.sqrt(data[i][1]**2+data[i][2]**2)*math.sqrt(data[j][1]**2+data[j][2]**2))
            avg_sim = avg_sim + cos_sim_calc
            avg_count = avg_count + 1


    avg_sim = avg_sim/avg_count
    print('---------------------------------------------------------------------')
    print('The average similarity between the documents are:', end = ' ')
    print(avg_sim)
    print('\n\n')


#testing:
print('Testing Values:')
compare_similarities_2docs(doc_1, doc_2)


#prepping movie reviews:
print('Movie Review Values:')
neg_review_ids = list(movie_reviews.fileids('neg'))
pos_review_ids = list(movie_reviews.fileids('pos'))

neg_review_doc = []
pos_review_doc = []

for i in range(len(neg_review_ids)):
    neg_review = movie_reviews.words(neg_review_ids[i])
    for j in range(len(neg_review)):
        neg_review_doc.append(neg_review[j])

for i in range(len(pos_review_ids)):
    pos_review = movie_reviews.words(pos_review_ids[i])
    for j in range(len(pos_review)):
        pos_review_doc.append(pos_review[j])

#real function application:
compare_similarities_2docs(neg_review_doc, pos_review_doc)
