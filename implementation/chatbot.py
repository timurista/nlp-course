# chatbot

import numpy as np
import tensorflow as tf
import re
import time

### Part 1 ###
lines = open('data/movie_lines.txt', 
             encoding='utf-8', 
             errors='ignore').read().split('\n')

conversations = open('data/movie_conversations.txt', 
             encoding='utf-8', 
             errors='ignore').read().split('\n')

print(lines[0:10])

# dict from id to line
id2line = {}

for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]
    

conversations_ids = []
for conversation in conversations[:-1]:
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ", "")
    conversations_ids.append(_conversation.split(','))
    
answers = []
questions = []
for conversation in conversations_ids:
    for i in range(len(conversation) - 1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i + 1]])
        
## Cleaning process, lowercase and remove apostrophes etc
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "were is", text)    
    text = re.sub(r"where's", "were is", text)    
    text = re.sub(r"\'ll", " will", text)    
    text = re.sub(r"\'ve", " have", text)               
    text = re.sub(r"\'re", " are", text)     
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)         
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)         
    return text

# Clean them now
clean_questions = []
for q in questions:
    clean_questions.append(clean_text(q))
    
clean_answers = []
for a in answers:
    clean_answers.append(clean_text(a))


## Remove non frequent words
word2count = {}
for q in clean_questions:
    for word in q.split():
        w = word.strip()
        if w not in word2count:
            word2count[w] = 1
        else:
            word2count[w]+= 1

for a in clean_answers:
    for word in a.split():
        w = word.strip()
        if w not in word2count:
            word2count[w] = 1
        else:
            word2count[w]+= 1

## remove words not at threshold
threshold = 20
questionswords2int = {}
word_number = 0

for word, count in word2count.items():
    if count >= threshold:
        questionswords2int[word] = word_number
        word_number += 1

answerswords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold:
        answerswords2int[word] = word_number
        word_number += 1        
        
## tokens for seq2seq
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
# adding +1 since we need unique number, and numbers go from 0 to len of words
# so we just use length of words dict here
for token in tokens:
    questionswords2int[token] = len(questionswords2int) + 1

for token in tokens:
    answerswords2int[token] = len(answerswords2int) + 1

## Create inversity dictionary of answerswords2int dictionary
answersint2word = { w_i: w for w, w_i in answerswords2int.items() }

for i in range(len(clean_answers)):
    clean_answers[i] += " <EOS>"

## Translating
questions_to_int = []
for question in clean_questions:
    ints = []
    for w in question.split():
        if w not in questionswords2int:
            ints.append(questionswords2int['<OUT>'])
        else:
            ints.append(questionswords2int[w])
    
    questions_to_int.append(ints)

    
answers_to_int = []
for answer in clean_answers:
    ints = []
    for w in answer.split():
        if w not in answerswords2int:
            ints.append(answerswords2int['<OUT>'])
        else:
            ints.append(answerswords2int[w])
    
    answers_to_int.append(ints)


# sorting questions and answers by lenght of questions

sorted_clean_questions = []
sorted_clean_answers = []

for length in range(1, 26):
    for i in enumerate(questions_to_int):
        if len(i[1]) == length:
            sorted_clean_questions.append(questions_to_int[i[0]])
            sorted_clean_answers.append(answers_to_int[i[0]])
            
            