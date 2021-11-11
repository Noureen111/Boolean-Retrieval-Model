### LIBRARIES

import re
import numpy as np
from nltk.stem import PorterStemmer
import pandas as pd

filename = "CS317-IR Dataset for A1/ShortStories/"
ps = PorterStemmer()



### READING STOP WORDS

stop_words = []
f = open("CS317-IR Dataset for A1/Stopword-List.txt", "r")
text = f.read().replace('\n', ' ')
f.close()
stop_words = text.split()




### FUNCTION FOR CONTRACTIONS

def decontract_words(text):

    # For general words
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"[0-9] +", "", text)
    text = re.sub(r"[^\w\s]", " ", text)


    # For specific words
    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"can\'t", "can not", text)

    return text



### INVERTED INDEX

def inverted_index(stop_words):

    dictionary = {}
    documents = {}
    filename = "CS317-IR Dataset for A1/ShortStories/"

    # For each document
    for i in range(1, 51):   

        document_no = i
        doc = str(document_no)
        f = open(filename + doc + ".txt", "r", encoding = "utf-8")
        text = f.read().replace('\n', ' ')
        f.close()
        text = decontract_words(text)
        
        file = doc 
        documents.setdefault(file, [])         
        documents[file].append(text)


        # Removing stopwords and converting to lowercase
        text = text.lower()
        text = [ w if w not in stop_words else '' for w in text.split(' ')]
        docs = []
        docs = list(filter(None, text))
        stemmed_words = []

        # stemming out
        for w in docs:
            stemmed_words.append(ps.stem(w))


        # creating posting list for every document
        for w in stemmed_words:
            term = w
            dictionary.setdefault(term, [])
            dictionary[term].append(document_no)    

        dictionary = { w1: list(set(w2)) for w1,w2 in dictionary.items() }

    return dictionary, documents



### POSITIONAL INDEX

def positional_index(stop_words):

    dictionary = {}
    documents = {}
    fileno = 1

    # For each document
    for i in range(1, 51):

        document_no = i
        doc = str(document_no)
        f = open(filename + doc + ".txt", "r", encoding = "utf-8")
        text = f.read().replace('\n', ' ')[1:]
        f.close()
        text = decontract_words(text)

        term = doc 
        documents.setdefault(term, [])
        documents[term].append(text)
        text = text.lower()
        text = text.split(' ')
        docs = []
        docs = list(filter(None, text))
        stemmed_words = []

        # stemming out
        for i in docs:
            stemmed_words.append(ps.stem(i))

        # creating positional  index posting lists
        for position, term in enumerate(stemmed_words):

            if term in dictionary:

                dictionary[term][0] = dictionary[term][0] + 1

                if document_no in dictionary[term][1]:
                    dictionary[term][1][document_no].append(position)

                else:
                    dictionary[term][1][document_no] = [position]

            else:
                dictionary[term] = []
                dictionary[term].append(1)
                dictionary[term].append({})
                dictionary[term][1][document_no] = [position]


    return dictionary,documents



### AND FUNCTION
def AND_operation(list1, list2):
    if ((list1) and (list2)):
        return set(list1).intersection(list2)
    
    else:
        return set()


### OR FUNCTION
def OR_operation(list1, list2):
    return set(list1).union(list2)


## NOT FUNCTION
def NOT_operation(list1):

    total_documents = [i for i in range(1, 51)]
    if list1 is None:
        return total_documents
    
    else:
        return set(total_documents).symmetric_difference(set(list1))    
    




## PROCESSING PROXIMITY QUERIES

def process_proximity_query(user_query, positional_dictionary):
    
    user_query = re.sub(r"AND", "", user_query)
    user_query = re.sub(r"and", "", user_query)
    user_query = re.sub(r" ", " ", user_query)
    user_query = user_query.split(' ')
    query = []

    for i in user_query:
        query.append(ps.stem(i))

    
    word1 = positional_dictionary.get(query[0])
    word1 = word1[1]
    word2 = positional_dictionary.get(query[1])
    word2 = word2[1]
    common_documents = set(word1).intersection(word2)

    query[2] = re.sub(r"/", "", query[2])
    result = []
    skip_words = int(query[2]) + 1


    for ii in common_documents:
        for w1 in word1[ii]:
            for w2 in word2[ii]:
                if(abs(w1 - w2) == skip_words):
                    result.append(ii)
    
    result = list(dict.fromkeys(result))
    return result




### POSTFIX FUNCTION

def postfix_query(tokens):

    # precedence of query operators
    prec = {}
    prec['NOT'] = 3
    prec['AND'] = 2
    prec['OR'] = 1
    prec['('] = 0
    prec[')'] = 0

    result = []
    op_stack = []

    # creating postfix expression
    for token in tokens:
        if (token == '('):
            op_stack.append(token)

        elif (token == ')'):
            op = op_stack.pop()
            while op != '(':
                result.append(op)
                op = op_stack.pop()

        elif (token in prec):
            if op_stack:
                curr_op = op_stack[-1]
                while (op_stack and prec[curr_op] > prec[token]):
                    result.append(op_stack.pop())
                    if (op_stack):
                        curr_op = op_stack[-1]
                    
            op_stack.append(token)

        else:
            result.append(token.lower())

    
    while op_stack:
        result.append(op_stack.pop())
    
    print("pstf" , result)

    return result




### PROCESSING BOOLEAN QUERIES

def process_query(user_query, inverted_dictionary):

    user_query = user_query.replace('(','( ')
    user_query = user_query.replace(')', ' )')
    user_query = user_query.split(' ')

    q = []

    for term in user_query:
        q.append(ps.stem(term))
    
    for i in range(0, len(q)):
        if (q[i] == 'and' or q[i] == 'or' or q[i] == 'not' ):

            q[i] = q[i].upper()
        
    resulted_stack = []
    postfixed = postfix_query(q)

    # Evaluating postfix query expression
    for term in postfixed:
        if (term != 'AND' and term != 'OR' and term != 'NOT'):
            term = term.replace('(', ' ')
            term = term.replace(')', ' ')
            term = term.lower()
            term = inverted_dictionary.get(term)
            resulted_stack.append(term)
        
        elif (term == 'AND'):
            x = resulted_stack.pop()
            y = resulted_stack.pop()
            resulted_stack.append(AND_operation(x, y))

        elif (term == 'OR'):
            x = resulted_stack.pop()
            y = resulted_stack.pop()
            resulted_stack.append(OR_operation(x, y))
    
        elif (term == 'NOT'):
            x = resulted_stack.pop()
            # if x is not None:
            resulted_stack.append(NOT_operation(x))
            # else:
                # pass
            # print("List gotten: ", x)


    return resulted_stack.pop()



### TAKING QUERY

query = input("Enter your query: ")
inverted_dictionary, documents = inverted_index(stop_words)

if '/' in query:
    positional_dictionary, documents = positional_index(stop_words)
    result = process_proximity_query(query, positional_dictionary)

else:
    query = decontract_words(query)
    query = query.lower()
    result = process_query(query, inverted_dictionary)




### FINAL RESULT

result = list(result)
result.sort()
print("Result: ", result)