# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
#https://www.kaggle.com/tanumoynandy/sarcasm-detection-rnn-lstm <-- HOME WORK
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D , Dropout , Conv1D , MaxPooling1D , Activation , GlobalMaxPooling1D , Input, Lambda, Dense
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Model
import keras.backend as K
from collections import Counter
import tensorflow_hub as hub
import tensorflow as tf
import sys
import re


# ClassifyIt = True

def read_csv(filepath):
    df_chunk = pd.read_csv(filepath)
    #df_chuck = df_chuck.fillna(0)
    return df_chunk

max_features = 240 #length of the maximum ones
if len(sys.argv[1]) > 1:
    df = read_csv(sys.argv[1])
else:
    print("Error - NO INPUT FILE GIVEN")
    exit()
if len(sys.argv[2]) > 1:
    nameThingy = sys.argv[2]
else:
    print("Error - NO OUTPUT FILE GIVEN")
    exit()
df['Cleaned'] = df['Cleaned'].str.lower()
totalNum = df['Cleaned'].str.len()
# calculate the padding sequence based on the max length/average length
avg = np.mean(totalNum)
max_len = int(avg)
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(df['Cleaned'].values)
X = tokenizer.texts_to_sequences(df['Cleaned'].values)
X = pad_sequences(X,maxlen=max_len)
print(X.shape[1])

url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
embed = hub.Module(url,trainable=True)

x = df['Cleaned']

with tf.Session() as session:
    K.set_session(session)
    session.run(tf.global_variables_initializer())  
    session.run(tf.tables_initializer())
    tweet_embeddings = embed(x.tolist())
    tweet_embeddings = tweet_embeddings.eval()
    #session.run(tweet_embeddings)
    #tweet_embeddings = np.array(tweet_embeddings)
    print("FUCKING SHAPE IS ", tweet_embeddings.shape)
    print(tweet_embeddings)
   
    embeeddingName = nameThingy + ".npy"
    np.save(embeeddingName,tweet_embeddings)

#Download the Model


def UniversalEmbedding(x):
    return embed(tf.squeeze(tf.cast(x, tf.string)))


# Y = df['Prediction'].values
# weights = get_class_weights(Y)	
# Y = to_categorical(Y, num_classes=2, dtype='float32')

# embed_dim = 50
# lstm_out = 64
# print(weights)




# #model = Sequential()
# #model.add(Embedding(input_dim=max_features, output_dim=embed_dim,input_length = X.shape[1]))
# #model.add(LSTM(embed_dim, dropout=0.2, recurrent_dropout=0.2))
# #model.add(Dense(32,activation='tanh'))
# #model.add(Dense(2,activation='softmax'))
# test = UniversalEmbedding(x)
# print(type(test))
# input_text = Input(shape=(1,), dtype=tf.string)
# embedding = Lambda(UniversalEmbedding, output_shape=(512, ))(input_text)
# #lgbt = LSTM(32,dropout=True, recurrent_dropout=0.4) (embedding)
# #lstm_encode = LSTM(16,recurrent_dropout=0.4) (lgbt)
# dense = Dense(256, activation='relu')(embedding)
# pred = Dense(2, activation='softmax')(dense)
# model = Model(inputs=[input_text], outputs=pred)
# model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
# print(model.summary())
# X_train, X_test, Y_train, Y_test = train_test_split(x,Y, test_size = 0.30, random_state = 42)
# batch_size = 64
# #model.fit(X_train, Y_train, epochs = 10, batch_size=batch_size, verbose = 2, class_weight=weights)
# #validation_size = 1500

# with tf.Session() as session:
#     K.set_session(session)
#     session.run(tf.global_variables_initializer())  
#     session.run(tf.tables_initializer())
#     tweet_embeddings = embed(x.tolist())
#     tweet_embeddings = tweet_embeddings.eval()
#     #session.run(tweet_embeddings)
#     #tweet_embeddings = np.array(tweet_embeddings)
#     print("FUCKING SHAPE IS ", tweet_embeddings.shape)
#     print(tweet_embeddings)
   

#     np.save('embeddings.npy',tweet_embeddings)

#     #history = model.fit(X_train, Y_train, epochs=25, batch_size=32, class_weight=weights)
#    # score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
#     #model.save_weights("model.h5")
#     #print("Saved model to disk")

# np.save('embeddings.npy',tweet_embeddings)


#with tf.Session() as session:
#    K.set_session(session)
#    session.run(tf.global_variables_initializer())
#    session.run(tf.tables_initializer())
#    model.load_weights('./model.h5')  
#    score,acc = model.evaluate(X_test, Y_test, verbose = 2)
#    print("score: %.2f" % (score))
#    print("acc: %.2f" % (acc))



#X_validate = X_test[-validation_size:]
#Y_validate = Y_test[-validation_size:]
#X_test = X_test[:-validation_size]
#Y_test = Y_test[:-validation_size]
#score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
#print("score: %.2f" % (score))
#print("acc: %.2f" % (acc))


# if ClassifyIt is True:
# 	df2 = read_csv2("test_noannotations.csv")
# 	df2['pred'] = df2['Comment'].apply(predictText)
# 	df2.to_csv('lstm_output.csv',index=False)
# 	#df3 = pd.DataFrame(tester)
# 	#df3.to_csv('output_lstm.csv',index=False)



