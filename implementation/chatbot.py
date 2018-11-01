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
            

## Part 2 Seq2Seq Model

def model_inputs():
    # input for tesnorflow input
    # sorted clean questions are integers
    # 2d matrix
    inputs = tf.placeholder(tf.int32, [None, None], name = 'input')    

    # target
    targets = tf.placeholder(tf.int32, [None, None], name = 'target')    
    
    # learning rate and param for dropout, how many nuerons
    # you deactive
    lr = tf.placeholder(tf.float32, name = 'learnin_rate')    

    # controls dropout rate
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')    

    return inputs, targets, lr, keep_prob

## Preprocessing the targets
## need to feed nueral network with answers in batches
## batches of size say 10
## need to put the SOS token at beginning f each answer in sorted_clean_answers

def preprocess_targets(targets, word2int, batch_size):
    left_side = tf.fill([batch_size, 1], word2int['<SOS>'])
    ## -1 gets all columns but the last size
    ## how many do we want to slice by?
    ## we want them all, so do slice [1,1]
    right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1])
    
    # horizontal access is 1
    preprocesssed_targets = tf.concat([left_side, right_side], 1)
    return preprocesssed_targets

## Encoding layer RNN
## LSTM
def encoder_rnn_layer(rnn_inputs, 
                      rnn_size, 
                      num_layers, 
                      keep_prob,
                      # length questions in each batch
                      sequence_length):
    
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size);
    # apply dropout iva keep_prob
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, 
                                                 input_keep_prob = keep_prob)
    
    # encoder step
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    
    # bidirectional takes input
    # built forward and backward rnn
    # input size of forward and backward cell matches
    # both directions on same
    _, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                                                       cell_bw = encoder_cell,
                                                       sequence_length = sequence_length,
                                                       inputs = rnn_inputs,
                                                       dtype = tf.float32)
    return encoder_state

## Decoder RNR layer
# takes encoder state we got from the encoder rnn layer
def decode_training_set(encoder_state,
                        decoder_cell,
                        decoder_embedded_input,
                        sequence_length,
                        ## variable scope
                        ## advanced data structure to wrap
                        decoder_scope,
                        output_function,
                        keep_prob,
                        batch_size
                        ):
    # number of lines or cols
    # number of elements in 3
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    ## seq2seq prepare attention
    
    ## attention keys to be compared with attention state
    ## attention values
    ## attention score, similarity between keys
    ## training decoder function
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, 
                                                                                                                                    attention_option= 'bahdanau',
                                                                                                                                    num_units = decoder_cell.output_size)

    ## attentional for future decoder
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              name = "attn_dec_train")
    ## returns 3 elements
    ## it returns decoder output
    ## decoder_final_state
    ## decoder_final_context_state
    decoder_output, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                  training_decoder_function,
                                                                  decoder_embedded_input,
                                                                  sequence_length,
                                                                  scope = decoder_scope)
    ## output to apply dropout
    ## and dropout rate
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    
    ## this then calls output function and returns results
    return output_function(decoder_output_dropout)


## Decoding the test/validation set
# new stuff not used for training
## TESTING PART / Validation
# Keep 10% of training data for improving accuracy
# we use inference function, deduce logically answers to questions it asked

def decode_test_set(encoder_state,
                    decoder_cell,
                    decoder_embeddings_matrix,
                    ## 4 new args
                    sos_id,
                    eos_id,
                    maximum_length,
                    num_words, # total number words
                    decoder_scope,
                    output_function,
                    keep_prob,
                    batch_size,                        
                    ):

    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, 
                                                                                                                                    attention_option= 'bahdanau',
                                                                                                                                    num_units = decoder_cell.output_size)

    ## attentional for future decoder
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embeddings_matrix,
                                                                              sos_id,
                                                                              eos_id,
                                                                              maximum_length,
                                                                              num_words,
                                                                              # namescope is mode of attentions fn
                                                                              name = "attn_dec_inf")
    ## returns 3 elements
    test_predictions, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                    test_decoder_function,
                                                                    scope = decoder_scope)

    ## this then calls output function and returns results
    return test_predictions


## decoder RNN
def decoder_rnn(decoder_embedded_input,
                decoder_embeddings_matrix,
                encoder_state,
                num_words,
                sequence_length,
                rnn_size,
                num_layers,
                word2int,
                keep_prob,
                batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        # lstm layer
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        ## initialize some weights for last layer
        weights = tf.truncated_normal_initializer(stddev = 0.1)
        biases = tf.zeros_initializer()
        ## last layer of rnn output
        ## fully connected layer at end
        output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                      num_words,
                                                                      None,
                                                                      scope = decoding_scope,
                                                                      weights_initializer = weights,
                                                                      biases_initializer = biases)
        training_predictions = decode_training_set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   sequence_length,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,
                    decoder_cell,
                    decoder_embeddings_matrix,
                    word2int['<SOS>'],
                    word2int['<EOS>'],
                    sequence_length - 1,
                    num_words,
                    decoding_scope,
                    output_function,
                    keep_prob,
                    batch_size
                    )
    return training_predictions, test_predictions
        

## Building seq2seq model
## targets = answers
def seq2seq_model(inputs, 
                  targets, 
                  keep_prob, 
                  batch_size, 
                  sequence_length, 
                  answers_num_words,
                  questions_num_words,
                  # number dimensions for encoder/decoder
                  encoder_embedding_size,
                  decoder_embedding_size,
                  rnn_size,
                  num_layers,
                  questionswords2int):
    ## return training and test predictions
    ## assemblage, return everything
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                              answers_num_words + 1,
                                                              encoder_embedding_size,
                                                              initializer = tf.random_uniform_initializer(0, 1))
    ## feed rnn
    encoder_state = encoder_rnn_layer(encoder_embedded_input, 
                                      rnn_size, 
                                      num_layers, 
                                      keep_prob,
                                      sequence_length)
    ## get preprocessed stuff
    ## to backpropogate the loss
    preprocessed_targets = preprocess_targets(targets, questionswords2int, batch_size)
    
    # matrix
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1, decoder_embedding_size], 0, 1))

    # use matrix to get decoder embedded inputs
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    
    ## now we can decode things
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input,
                                                         decoder_embeddings_matrix,
                                                         encoder_state,
                                                         questions_num_words,
                                                         sequence_length,
                                                         rnn_size,
                                                         num_layers,
                                                         questionswords2int,
                                                         keep_prob,
                                                         batch_size)
    
    return training_predictions, test_predictions
    
    