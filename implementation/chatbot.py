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
    

    
################## PART 3 - TRAINING THE SEQ2SEQ MODEL ##################

## Hypoer params, epoch is whole process of forward propogation, then
## backpropogating loss and updating weights, one whole iteration of training
epochs = 100
batch_size = 64    
rnn_size = 512
num_layers = 3
encoding_embedding_size = 512 #512 cols in embedding matrix
decoding_embedding_size = 512
learning_rate = 0.01 # can't be too high, or too low
learning_rate_decay = 0.9 # percentage it is decayed so it learns in more depth    
min_learning_rate = 0.0001
keep_probability = 0.5

## Defining session
# when you open in gneeral you have to reset graph
tf.reset_default_graph()
session = tf.InteractiveSession()

## loading models
inputs, targets, lr, keep_prob = model_inputs()

## set sequence length
sequence_length = tf.placeholder_with_default(25, None, name = "sequence_length" )

# getting the shape of the inputs tensor
input_shape = tf.shape(inputs)

# get the training and test predictions
# we reverse dimensions of a tensor
training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs, [-1]),
                                                       targets,
                                                       keep_prob,
                                                       batch_size,
                                                       sequence_length,
                                                       len(answerswords2int),
                                                       len(questionswords2int),
                                                       encoding_embedding_size,
                                                       decoding_embedding_size,
                                                       rnn_size,
                                                       num_layers,
                                                       questionswords2int
                                                       )

## Setup Loss Error

# gradient clipping to avoid vanishing gradient

# loss error is weighted entropy
# atom optimizer for stochastic gradient descent
with tf.name_scope("optimization"):
    loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions,
                                                  targets,
                                                  tf.ones([input_shape[0], sequence_length]))
    # atom optimizer then clipping
    # then apply gradient clipping to optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # optimizer to compute gradient
    gradients = optimizer.compute_gradients(loss_error)
    clipped_gradients = [(tf.clip_by_value(grad_tensor, -5.0, 5.0), grad_variable) for grad_tensor, grad_variable in gradients if grad_tensor is not None ]
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)
    

## Padding seuences so question and answers same length
def apply_padding(batch_of_sequences, word2int):
    max_sequence_length = max([ len(sequence) for sequence in batch_of_sequences ])
    return [ sequence + [word2int['<PAD>']] * (max_sequence_length - len(sequence)) for sequence in batch_of_sequences]


## make the batches of questions and answers
    
def split_into_batches(questions, answers, batch_size):
    # // gives us integer
    for batch_index in range(0, len(questions) // batch_size):
        start_index = batch_index * batch_size
        questions_in_batch = questions[start_index : start_index + batch_size]
        answers_in_batch = answers[start_index : start_index + batch_size]
        # need numpy arry for this
        # so we got to convert padding lists to arrays
        padded_questions_in_batch = np.array(apply_padding(questions_in_batch, questionswords2int))
        padded_answers_in_batch = np.array(apply_padding(answers_in_batch, answerswords2int))
        yield padded_questions_in_batch, padded_answers_in_batch
    
## splitting training and validation sets
## observations testing model on side to see how it does on new observation
## 10-15% of data for validation

## index of 15% and last 85%
training_validation_split = int(len(sorted_clean_questions) * 0.15 )
training_questions = sorted_clean_questions[training_validation_split:]
training_answers = sorted_clean_answers[training_validation_split:]

validation_questions = sorted_clean_questions[:training_validation_split]
validation_answers = sorted_clean_answers[:training_validation_split]

## Training
batch_index_check_training_loss = 100

## training halfway
batch_index_check_validation_loss = (len(training_questions) // batch_size // 2) - 1
# 
total_training_loss_error = 0
total_validation_loss_error = 0

# list to check if we reach loss below min we get
list_validation_loss_error = []
# check each time we don't reduce validation loss
# we stop training when it reaches a number
early_stopping_check = 0
early_stopping_stop = 1000
checkpoint = "chatbot_weights.ckpt" #file w weights

session.run(tf.global_variables_initializer())

for epoch in range(1, epochs + 1):
    # have to go back and forth in model
    for batch_index, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(training_questions, training_answers, batch_size)):
        starting_time = time.time()
        # need loss error
        _, batch_training_loss_error = session.run([optimizer_gradient_clipping, loss_error], { inputs: padded_questions_in_batch,
                                                   targets:padded_answers_in_batch,
                                                   lr: learning_rate,
                                                   sequence_length: padded_answers_in_batch.shape[1],
                                                   keep_prob: keep_probability
                                                   })
        total_training_loss_error += batch_training_loss_error
        ending_time = time.time()
        batch_time = ending_time - starting_time
        if batch_index % batch_index_check_training_loss == 0:
            print('Epoch: {:>3}/{}, Batch: {:>4}/{}, Training Loss Error: {:>6.3f}, Training Time on 100 Batches: {:d} seconds'.format(epoch,
                                                                                                                                       epochs,
                                                                                                                                       batch_index,
                                                                                                                                       len(training_questions) // batch_size,
                                                                                                                                       total_training_loss_error / batch_index_check_training_loss,
                                                                                                                                      int(batch_time * 100) ))
            total_training_loss_error = 0
        
        ## validation batch
        if batch_index % batch_index_check_validation_loss == 0 and batch_index > 0:
            ## initialize loss erro
            total_validation_loss_error = 0
            starting_time = time.time()
            for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(validation_questions, validation_answers, batch_size)):
                batch_validation_loss_error = session.run(loss_error, {inputs: padded_questions_in_batch,
                                                                       targets:padded_answers_in_batch,
                                                                       lr: learning_rate,
                                                                       sequence_length: padded_answers_in_batch.shape[1],
                                                                       ## no need to activate they should be in set the nuerons
                                                                       keep_prob: 1})
                total_validation_loss_error += batch_training_loss_error
            
            ending_time = time.time()
            batch_time = ending_time - starting_time
            average_validation_loss_error = total_validation_loss_error // (len(validation_questions) // batch_size)
            print('Validation Loss Error: {:>6.3f}, Batch Validation Time: {:d} seconds'.format(average_validation_loss_error,batch_time))
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
            list_validation_loss_error.append(average_validation_loss_error)
            if average_validation_loss_error < min(list_validation_loss_error):
                print('I speak better now!!')
                early_stepping_check = 0
                saver = tf.train.Saver()
                saver.save(session, checkpoint)
            else:
                print('Sorry I do not speak better, I need to practice more.')
                early_stopping_check += 1
                if early_stopping_check == early_stopping_stop:
                    break
    
    if early_stopping_check == early_stopping_stop:
        print('My appologies this is the best I can do.')

print("Game Over")

####### Part 4 - Loading seq2seq model

checkpoint = "./chatbot_weights.ckpt"
session = tf.InteractiveSession()

session.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(session, checkpoint)

## Convert questions from strings to lists of encoded ints
def convert_string2int(question, word2int):
    question = clean_text(question)
    ## using get so out will be returned
    return [word2int.get(word, word2int['<OUT>']) for word in question.split()]

## setup the chat
while(True):
    question = input("You: ")
    if question == 'exit' or question == 'Goodbye'
        break
    question = convert_string2int(question, questionswords2int)
    question = question + [questionswords2int['<PAD>']] * (20 - len(question))
    
    ## nueral network is 20 questions
    fake_batch = np.zeros((batch_size, 20))
    fake_batch[0] = question
    predicted_answer = session.run(test_predictions, { inputs: fake_batch, keep_prob: 0.5 })[0]    
    answer = ''

    # get token ids from predicted answer
    for i in np.argmax(predicted_answer, 1):
        if answersint2word[i] == 'i':            
            token = 'I'
        elif answersint2word[i] == '<EOS>':
            token = '.'
        elif answersint2word[i] == '<OUT>':
            token = 'out'
        else:
            token = ' ' + answersint2word[i]
        answer += token
        if token == '.':
            # chatbot is done talking
            break
    print("ChatBot: " + answer)