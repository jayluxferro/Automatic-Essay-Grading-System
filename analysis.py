"""
Author: Jay Lux Ferro
Date:   18 Dec 2019
Task:   Analysis of results
"""
## imports
import db
import numpy as np
import matplotlib.pyplot as plt


## @definitions
models = [ { 'rnn': 'SimpleRNN + SimpleRNN' }, { 'lstm': 'LSTM + LSTM' }, { 'gru': 'GRU + GRU' }, { 'lstm_gru': 'LSTM + GRU' }, { 'gru_lstm': 'GRU + LSTM' }, { 'rnn_lstm': 'RNN + LSTM' }, { 'lstm_rnn': 'LSTM + SimpleRNN' }, { 'rnn_gru': 'RNN + SimpleRNN' }, { 'gru_rnn': 'GRU + SimpleRNN' } ]

embeddings = ['1', '2'] # embeddings.w2v.txt, word2vecmodel.bin

folds = [3, 4, 5] # K-Folds cross validation



emb1 = [None] * len(models)
emb2 = [None] * len(models)
emb1_counter = 0
emb2_counter = 0

emb1_names = []
emb2_names = []
for emb in embeddings:
    for m in models:
        for m_db_name, m_graph_name in m.items():
            avg = []
            for k in folds:
                avg.append(db.getAvg(emb, m_db_name, k))

            if emb is '1':
                emb1[emb1_counter] = avg
                emb1_counter = emb1_counter + 1
                emb1_names.append(m_graph_name)
            else:
                emb2[emb2_counter] = avg
                emb2_counter = emb2_counter + 1
                emb2_names.append(m_graph_name)


# Emb 1
counter = 0
print("\n #### Emb 1 ####")
plt.figure()
for e in emb1:
    print(emb1_names[counter], e, np.average(e))
    plt.plot(folds, e, '-o', label=emb1_names[counter])
    counter = counter + 1
plt.legend()
plt.xlabel('Folds')
plt.ylabel('QWK (%)')
#plt.title('Average QWK using K-Fold Cross Validation and Word Embeddings 1')
plt.show()


# Emb 2
counter = 0
plt.figure()
print("\n #### Emb 2 ####")
for e in emb2:
    print(np.max(e))
    print(emb2_names[counter], e, np.average(e))
    plt.plot(folds, e, '-o', label=emb2_names[counter])
    counter = counter + 1
plt.legend()
plt.xlabel('Folds')
plt.ylabel('QWK (%)')
#plt.title('Average QWK using K-Fold Cross Validation and Word Embeddings 2')
plt.show()
