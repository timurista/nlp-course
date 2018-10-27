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