"""
Author: Jay Lux Ferro
Date:   12 Dec 2018
Task:   AEGS
"""

### definitions ####
dataset_dir = './data'
output_dir = './output_dir'

### imports     ####
import os
import pandas as pd
import gensim.models as gm
from sklearn.cross_validation import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import cohen_kappa_score
from gensim.models import Word2Vec
import utils as utl
import models as mdl
import numpy as np
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
import db

X = pd.read_csv(os.path.join(dataset_dir, 'training_set_rel3.tsv'), sep='\t', encoding='ISO-8859-1')
y = X['domain1_score']
X = X.dropna(axis=1)
X = X.drop(columns=['rater1_domain1', 'rater2_domain1'])

## All models 
all_models = [ { 'rnn': mdl.simpleRNN2 }, { 'lstm': mdl.lstm2 }, { 'gru': mdl.gru2 }, { 'lstm_gru': mdl.lstm_gru2 }, { 'gru_lstm': mdl.gru_lstm2 }, { 'rnn_lstm': mdl.simpleRNN_lstm2 }, { 'lstm_rnn': mdl.lstm_simpleRNN2 }, { 'rnn_gru': mdl.simpleRNN_gru2 }, { 'gru_rnn': mdl.gru_simpleRNN2 } ]

# Initializing variables for word2vec model.
num_features = 300
min_word_count = 40
num_workers = 16
context = 10
downsampling = 1e-3

batch_size = 64
epochs = 100


# word embeddings
w1 = gm.KeyedVectors.load_word2vec_format('embeddings.w2v.txt')
w2 = gm.KeyedVectors.load_word2vec_format('word2vecmodel.bin', binary=True)

word_embeddings = [ {'1': w1 }, { '2': w2 } ]

# folds
folds = [3, 4, 5]


def process_data(model, emb_name, md, model_name, folds):
    cv = KFold(len(X), n_folds=folds, shuffle=True)
    results = []
    y_pred_list = []

    count = 1


    for traincv, testcv in cv:
        print("\n--------Fold {}--------\n".format(count))
        X_test, X_train, y_test, y_train = X.iloc[testcv], X.iloc[traincv], y.iloc[testcv], y.iloc[traincv]

        train_essays = X_train['essay']
        test_essays = X_test['essay']

        sentences = []

        for essay in train_essays:
                # Obtaining all sentences from the training essays.
                sentences += utl.essay_to_sentences(essay, remove_stopwords = True)

        print("Training Word2Vec Model...")
        
        clean_train_essays = []

        # Generate training and testing data word vectors.
        for essay_v in train_essays:
            clean_train_essays.append(utl.essay_to_wordlist(essay_v, remove_stopwords=True))
        trainDataVecs = utl.getAvgFeatureVecs(clean_train_essays, model, num_features)

        clean_test_essays = []
        for essay_v in test_essays:
            clean_test_essays.append(utl.essay_to_wordlist( essay_v, remove_stopwords=True ))
        testDataVecs = utl.getAvgFeatureVecs( clean_test_essays, model, num_features )

        trainDataVecs = np.array(trainDataVecs)
        testDataVecs = np.array(testDataVecs)
        
        # Reshaping train and test vectors to 3 dimensions. (1 represents one timestep)
        trainDataVecs = np.reshape(trainDataVecs, (trainDataVecs.shape[0], 1, trainDataVecs.shape[1]))
        testDataVecs = np.reshape(testDataVecs, (testDataVecs.shape[0], 1, testDataVecs.shape[1]))

        rnn_model = md(num_features)

        plot_model(rnn_model, to_file=output_dir + '/{}.eps'.format(model_name))
        rnn_model.fit(trainDataVecs, y_train, batch_size=batch_size, epochs=epochs)
        
        y_pred = rnn_model.predict(testDataVecs)


        # Round y_pred to the nearest integer.
        y_pred = np.around(y_pred)

        # Evaluate the model on the evaluation metric. "Quadratic mean averaged Kappa"
        result = cohen_kappa_score(y_test.values,y_pred,weights='quadratic')
        print("Kappa Score: {}".format(result))
        results.append(result)
      
        # Save weights
        weight_name = emb_name + "_" + model_name + "_" + str(folds) + "_" + str(count) + ".h5" 
        rnn_model.save('./model_weights/{}'.format(weight_name))

        # add data to db
        db.add_data(emb_name, model_name, folds, count, result, weight_name)

        count += 1

    print("Average Kappa score after a 5-fold cross validation: ", np.around(np.array(results).mean(),decimals=4))


def kill():
    sys.exit()

if __name__ == "__main__":
    # reset db
    db.reset()

    # start data processing
    for w in word_embeddings: # embeddings
        for emb_name, emb in w.items():
            for m in all_models: # models
                for m_name, m_instance in m.items():
                    for f in folds: # folds
                        num_features = 50 if emb_name == '1' else 300
                        process_data(emb, emb_name, m_instance, m_name, f) 
