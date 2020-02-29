import numpy as np
import os
import pandas as pd
import re



news_category = ["business", "entertainment", "politics", "sport", "tech"]


row_doc= "/Users/hp/Desktop/BBC News Summary/News Articles/"
summary_doc= "/Users/hp/Desktop/BBC News Summary/Summaries/"

data={"articles":[], "summaries":[]}

directories = {"news": row_doc, "summary": summary_doc}
row_dict = {}
sum_dict = {}

for path in directories.values():
    if path == row_doc:
        file_dict = row_dict
    else:
        file_dict = sum_dict
    dire = path
    for cat in news_category:
        category = cat
        files = os.listdir(dire + category)
        file_dict[cat] = files
        
        
row_data = {}
for cat in row_dict.keys():
    cat_dict = {}
    # row_data_frame[cat] = []
    for i in range(0, len(row_dict[cat])):
        filename = row_dict[cat][i]
        path = row_doc + cat + "/" + filename
        with open(path, "rb") as f:                
            text = f.read()
            cat_dict[filename[:3]] = text
    row_data[cat] = cat_dict
    
    
    
    
sum_data = {}
for cat in sum_dict.keys():
    cat_dict = {}
    # row_data_frame[cat] = []
    for i in range(0, len(sum_dict[cat])):
        filename = sum_dict[cat][i]
        path = summary_doc + cat + "/" + filename
        with open(path, "rb") as f:                
            text = f.read()
            cat_dict[filename[:3]] = text
    sum_data[cat] = cat_dict
    
    
news_business = pd.DataFrame.from_dict(row_data["business"], orient="index", columns=["row_article"])
news_business.head()



#  news_category = ["business", "entertainment", "politics", "sport", "tech"]
news_entertainment = pd.DataFrame.from_dict(row_data["entertainment"], orient="index", columns=["row_article"])
news_politics = pd.DataFrame.from_dict(row_data["politics"], orient="index", columns=["row_article"])
news_sport = pd.DataFrame.from_dict(row_data["sport"], orient="index", columns=["row_article"])
news_tech = pd.DataFrame.from_dict(row_data["tech"], orient="index", columns=["row_article"])



# summary data
summary_business = pd.DataFrame.from_dict(sum_data["business"], orient="index", columns=["summary"])
summary_entertainment = pd.DataFrame.from_dict(sum_data["entertainment"], orient="index", columns=["summary"])
summary_politics = pd.DataFrame.from_dict(sum_data["politics"], orient="index", columns=["summary"])
summary_sport = pd.DataFrame.from_dict(sum_data["sport"], orient="index", columns=["summary"])
summary_tech = pd.DataFrame.from_dict(sum_data["tech"], orient="index", columns=["summary"])




summary_business.head()


business = news_business.join(summary_business, how='inner')
entertainment = news_entertainment.join(summary_entertainment, how='inner')
politics = news_politics.join(summary_politics, how='inner')
sport = news_sport.join(summary_sport, how='inner')
tech = news_tech.join(summary_tech, how='inner')




business.head()



print("row", len(business.iloc[0,0]))
print("sum", len(business.iloc[0,1]))




list_df = [business, entertainment, politics, sport, tech]
length = 0
for df in list_df:
    length += len(df)
    
    
    
print("length of all data: ", length)


bbc_df = pd.concat([business, entertainment, politics, sport, tech], ignore_index=True)
len(bbc_df)













#TEXT CLEANING


def cleantext(text):
    text = str(text)
    text=text.split()
    words=[]
    for t in text:
        if t.isalpha():
            words.append(t)
    text=" ".join(words)
    text=text.lower()
    text=re.sub(r"what's","what is ",text)
    text=re.sub(r"it's","it is ",text)
    text=re.sub(r"\'ve"," have ",text)
    text=re.sub(r"i'm","i am ",text)
    text=re.sub(r"\'re"," are ",text)
    text=re.sub(r"n't"," not ",text)
    text=re.sub(r"\'d"," would ",text)
    text=re.sub(r"\'s","s",text)
    text=re.sub(r"\'ll"," will ",text)
    text=re.sub(r"can't"," cannot ",text)
    text=re.sub(r" e g "," eg ",text)
    text=re.sub(r"e-mail","email",text)
    text=re.sub(r"9\\/11"," 911 ",text)
    text=re.sub(r" u.s"," american ",text)
    text=re.sub(r" u.n"," united nations ",text)
    text=re.sub(r"\n"," ",text)
    text=re.sub(r":"," ",text)
    text=re.sub(r"-"," ",text)
    text=re.sub(r"\_"," ",text)
    text=re.sub(r"\d+"," ",text)
    text=re.sub(r"[$#@%&*!~?%{}()]"," ",text)
    
    return text



for col in bbc_df.columns:
    bbc_df[col] = bbc_df[col].apply(lambda x: cleantext(x))
    
    

bbc_df.head()




df.head()


len_list =[]
for article in df.row_article:
    words = article.split()
    length = len(words)
    len_list.append(length)
max(len_list)


#tokenizer

#bbc_df.to_csv(r'cleaned_bbc_news.csv')

    
    
    





  
bbc_art_sum = pd.read_csv("cleaned_bbc_news.csv")
bbc_art_sum.drop("Unnamed: 0", axis=1, inplace=True)
bbc_art_sum.head()


articles = list(bbc_art_sum.row_article)
summaries = list(bbc_art_sum.summary)




#tokeniz

from keras.preprocessing.text import Tokenizer
VOCAB_SIZE = 1500
tokenizer = Tokenizer(num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(articles)
article_sequences = tokenizer.texts_to_sequences(articles)
art_word_index = tokenizer.word_index
len(art_word_index)




print(article_sequences[0][:20])
print(article_sequences[1][:20])
print(article_sequences[2][:20])



art_word_index_1500 = {}
counter = 0
for word in art_word_index.keys():
    if art_word_index[word] == 0:
        print("found 0!")
        break
    if art_word_index[word] > VOCAB_SIZE:
        continue
    else:
        art_word_index_1500[word] = art_word_index[word]
        counter += 1
        
        
        
tokenizer.fit_on_texts(summaries)
summary_sequences = tokenizer.texts_to_sequences(summaries)
sum_word_index = tokenizer.word_index
len(sum_word_index)






sum_word_index_1500 = {}
counter = 0
for word in sum_word_index.keys():
    if sum_word_index[word] == 0:
        print("found 0!")
        break
    if sum_word_index[word] > VOCAB_SIZE:
        continue
    else:
        sum_word_index_1500[word] = sum_word_index[word]
        counter += 1




from keras.preprocessing.sequence import pad_sequences
MAX_LEN = 1000
pad_art_sequences = pad_sequences(article_sequences, maxlen=MAX_LEN, padding='post', truncating='post')


print(len(article_sequences[1]), len(pad_art_sequences[1]))

pad_sum_sequences = pad_sequences(summary_sequences, maxlen=MAX_LEN, padding='post', truncating='post')

print(len(summary_sequences[1]), len(pad_sum_sequences[1]))

#pad_art_sequences.shape


#pad_art_sequences.shape


#pad_art_sequences

encoder_inputs = np.zeros((2225, 1000), dtype='float32')
encoder_inputs.shape

decoder_inputs = np.zeros((2225, 1000), dtype='float32')
decoder_inputs.shape

for i, seqs in enumerate(pad_art_sequences):
    for j, seq in enumerate(seqs):
        encoder_inputs[i, j] = seq
        
for i, seqs in enumerate(pad_sum_sequences):
    for j, seq in enumerate(seqs):
        decoder_inputs[i, j] = seq


decoder_outputs = np.zeros((2225, 1000, 1500), dtype='float32')
decoder_outputs.shape


for i, seqs in enumerate(pad_sum_sequences):
    for j, seq in enumerate(seqs):
        decoder_outputs[i, j, seq] = 1.
        
        
        
        
        
        
        
embeddings_index = {}
with open('glove.6B.200d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

print('Found %s word vectors.' % len(embeddings_index))


def embedding_matrix_creater(embedding_dimention, word_index):
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dimention))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
          # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


art_embedding_matrix = embedding_matrix_creater(200, word_index=art_word_index_1500)
art_embedding_matrix.shape


sum_embedding_matrix = embedding_matrix_creater(200, word_index=sum_word_index_1500)
sum_embedding_matrix.shape

from keras.layers import Embedding
encoder_embedding_layer = Embedding(input_dim = 1500, 
                                    output_dim = 200,
                                    input_length = MAX_LEN,
                                    weights = [art_embedding_matrix],
                                    trainable = False)


decoder_embedding_layer = Embedding(input_dim = 1500, 
                                    output_dim = 200,
                                    input_length = MAX_LEN,
                                    weights = [sum_embedding_matrix],
                                    trainable = False)

sum_embedding_matrix.shape















from numpy.random import seed
seed(1)

from sklearn.model_selection import train_test_split
import logging

import plotly as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import pandas as pd
import pydot


import keras
from keras import backend as k
k.set_learning_phase(1)
from keras.preprocessing.text import Tokenizer
from keras import initializers
from keras.optimizers import RMSprop
from keras.models import Sequential,Model
from keras.layers import Dense,LSTM,Dropout,Input,Activation,Add,concatenate, Embedding, RepeatVector
from keras.layers.advanced_activations import LeakyReLU,PReLU
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam


from keras.layers import TimeDistributed






#############SIMPLE lstm ENCODER DECODER#################################


# encoder
encoder_inputs = Input(shape=(MAX_LEN, ), dtype='int32',)
encoder_embedding = encoder_embedding_layer(encoder_inputs)
encoder_LSTM = LSTM(HIDDEN_UNITS)(encoder_embedding)
# decoder
decoder_inputs = Input(shape=(MAX_LEN, ))
decoder_embedding = decoder_embedding_layer(decoder_inputs)
decoder_LSTM = LSTM(200)(decoder_embedding)
# merge
merge_layer = concatenate([encoder_LSTM, decoder_LSTM])
decoder_outputs = Dense(units=VOCAB_SIZE+1, activation="softmax")(merge_layer) # SUM_VOCAB_SIZE, sum_embedding_matrix.shape[1]

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.summary()


############# END OF SIMPLE lstm ENCODER DECODER#################################
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])





############# Bidirectional LSTM Encoder-Decoder-seq2seQ #################################


"""
Bidirectional LSTM: Others Inspired Encoder-Decoder-seq2seq
"""

encoder_inputs = Input(shape=(MAX_LEN,))
encoder_embedding = encoder_embedding_layer(encoder_inputs)
encoder_LSTM = LSTM(HIDDEN_UNITS, return_state=True)
encoder_LSTM_R = LSTM(HIDDEN_UNITS, return_state=True, go_backwards=True)
encoder_outputs_R, state_h_R, state_c_R = encoder_LSTM_R(encoder_embedding)
encoder_outputs, state_h, state_c = encoder_LSTM(encoder_embedding)

final_h = Add()([state_h, state_h_R])
final_c = Add()([state_c, state_c_R])
encoder_states = [final_h, final_c]

"""
decoder
"""
decoder_inputs = Input(shape=(MAX_LEN,))
decoder_embedding = decoder_embedding_layer(decoder_inputs)
decoder_LSTM = LSTM(HIDDEN_UNITS, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_LSTM(decoder_embedding, initial_state=encoder_states) 
decoder_dense = Dense(VOCAB_SIZE, activation='linear')
decoder_outputs = decoder_dense(decoder_outputs)

model= Model(inputs=[encoder_inputs,decoder_inputs], outputs=decoder_outputs)

model.summary()
############# END OF  Bidirectional LSTM Encoder-Decoder-seq2seQ #################################
rmsprop = RMSprop(lr=0.01, clipnorm=1.)
model.compile(loss='mse', optimizer=rmsprop, metrics=["accuracy"])




import numpy as np
num_samples = len(pad_sum_sequences)
decoder_output_data = np.zeros((num_samples, MAX_LEN, VOCAB_SIZE), dtype="int32")



for i, seqs in enumerate(pad_sum_sequences):
    for j, seq in enumerate(seqs):
        if j > 0:
            decoder_output_data[i][j][seq] = 1




art_train, art_test, sum_train, sum_test = train_test_split(pad_art_sequences, pad_sum_sequences, test_size=0.2)

train_num = art_train.shape[0]

target_train = decoder_output_data[:train_num]
target_test = decoder_output_data[train_num:]



history = model.fit([art_train, sum_train], 
                     target_train, 
                     epochs=EPOCHS, 
                     batch_size=BATCH_SIZE,
                     validation_data=([art_test, sum_test], target_test))








####visualization


import matplotlib.pyplot as plt
#%matplotlib inline

plt.figure(figsize=(10, 6))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#%matplotlib inline
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()




plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
plt.show()


model_json = model.to_json()
with open('text_summary.json', "w") as json_file:
    json_file.write(model_json)

model.save_weights("text_summary.h5")
model.load_weights('text_summary.h5')
print("Saved Model!")



reverse_target_word_index=y_tokenizer.index_word
reverse_source_word_index=x_tokenizer.index_word
target_word_index=y_tokenizer.word_index



MAX_INPUT_LENGTH = 1000
VOCAB_SIZE =1500
EMBEDDING_DIM = 200
HIDDEN_UNITS_ENC = 200
#VOCAB_SIZE=VOCAB_SIZE+1


LEARNING_RATE = 0.002
BATCH_SIZE = 32
EPOCHS = 5



encoder_inputs = Input(shape=(MAX_INPUT_LENGTH,), name='encoder_inputs')
embedding_layer = Embedding(1500 + 1, EMBEDDING_DIM, weights=[art_embedding_matrix],
                            input_length=MAX_INPUT_LENGTH, trainable=False, name='embedding_layer')

encoder_rnn = LSTM(units=HIDDEN_UNITS_ENC, return_state=True, dropout=0.5, recurrent_dropout=0.5,name='encoder_lstm')
encoder_output, state_h_f, state_c_f = encoder_rnn(embedding_layer(encoder_inputs))
encoder_rnn2 = LSTM(units=HIDDEN_UNITS_ENC, return_state=True, dropout=0.5, recurrent_dropout=0.5,
go_backwards=True,name='encoder_lstm_backward')
encoder_output, state_h_b, state_c_b = encoder_rnn2(embedding_layer(encoder_inputs))

state_h = concatenate([state_h_f, state_h_b])
state_c = concatenate([state_c_f, state_c_b])

encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(MAX_INPUT_LENGTH,), name='decoder_inputs')
embedding_layer = Embedding(1500 + 1, EMBEDDING_DIM, weights=[sum_embedding_matrix],
input_length=MAX_INPUT_LENGTH, trainable=False, name='emb_2')
decoder_lstm = LSTM(HIDDEN_UNITS_ENC * 2, return_sequences=False, return_state=True, dropout=0.5,
recurrent_dropout=0.5, name='decoder_lstm')
decoder_outputs, state_h, state_c = decoder_lstm(embedding_layer(decoder_inputs), initial_state=encoder_states)

decoder_dense = Dense(200, name='decoder_dense')
decoder_outputs = decoder_dense(decoder_outputs)
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)




def decode_sequence(input_seq):
    # Encode the input as state vectors.
    e_out, e_h, e_c = model.predict(input_seq)
    
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    
    # Populate the first word of target sequence with the start word.
    target_seq[0, 0] = target_word_index['sostok']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
      
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]
        
        if(sampled_token!='eostok'):
            decoded_sentence += ' '+sampled_token

        # Exit condition: either hit max length or find stop word.
        if (sampled_token == 'eostok'  or len(decoded_sentence.split()) >= (max_summary_len-1)):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence


def seq2summary(input_seq):
    newString=''
    for i in input_seq:
        newString=newString+reverse_target_word_index[i]+' '
    return newString

def seq2text(input_seq):
    newString=''
    for i in input_seq:
        if(i!=0):
            newString=newString+reverse_source_word_index[i]+' '
    return newString


print("Original summary:",seq2summary(y_tr[i]))
print("Predicted summary:",decode_sequence(x_tr[i].reshape(1,max_text_len)))
print("\n")


