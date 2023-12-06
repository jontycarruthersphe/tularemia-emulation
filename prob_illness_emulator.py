# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 08:56:58 2023

@author: Jonathan.Carruthers
"""

import time
import numpy as np
import pandas as pd
from tensorflow import keras
from pathlib import Path

'''
Use 10-fold cross validation to identify a model that can accurately predict
the probability of response. Creates a .csv file with a summary of model
performance for different architectures.
'''

# load the training data
data = np.load('training_data/prob_illness.npy')
np.random.shuffle(data)


def performance(y_pred, y_true, tol=0.005):
    '''
    Calculate the the fraction of predictions that lie within a given tolerance
    of the true values
    '''
    return (np.abs(y_pred.flatten() - y_true) < tol).sum() / y_true.shape[0]


def train_model(x_train, y_train, x_test=None, y_test=None, save_model=False):
    # define the model
    model = keras.Sequential()
    model.add(keras.layers.Dense(N_HIDDEN, batch_input_shape=(None, 3), 
                                 activation='relu'))

    for _ in range(N_LAYERS-1):
        model.add(keras.layers.Dense(N_HIDDEN, activation='relu'))
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    # compile the model
    model.compile(loss="mean_squared_error", 
                  optimizer=keras.optimizers.Adam(learning_rate=lr))
    
    # fit the model
    model.fit(x=x_train, y=y_train, batch_size=batch_size, 
              epochs=epochs, verbose=0)
    
    # save the model if necessary
    if save_model:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        model.save(path / f'model_NLAYERS={N_LAYERS}_NHIDDEN={N_HIDDEN}_{timestr}.h5')
    
    # evaluate the model using the training and test folds
    train_performance = performance(model.predict(x_train), y_train)
    if x_test is None:
        return train_performance
    
    test_performance = performance(model.predict(x_test), y_test)
    return train_performance, test_performance
    

lr = 1e-3           # learning rate
epochs = 100
batch_size = 25

path = Path('prob_illness_models')
path.mkdir(exist_ok=True)


# run the k-fold cross validation
n_folds = 10
fold_size = int(data.shape[0]/n_folds) + 1


model_summary = pd.DataFrame()
for N_LAYERS in [2, 3, 4, 5]:
    for N_HIDDEN in [64, 128, 256]:

        model_performance = np.zeros((n_folds, 2))

        for k in range(n_folds):
            test = data[k*fold_size:(k+1)*fold_size]
            train = np.vstack((data[:k*fold_size], data[(k+1)*fold_size:]))
        
            # check that none of the test sets are in the training set
            for row in test: assert not ((row==train).all(axis=1)).any()
            
            x_train, y_train = train[:,:-1], train[:,-1]
            x_test, y_test = test[:,:-1], test[:,-1]
            
            train_p, test_p = train_model(x_train, y_train, x_test, y_test)
            
            model_performance[k] = [train_p, test_p]
               
        # add to the summary of models
        summary_vals = np.array([N_LAYERS, N_HIDDEN, np.mean(model_performance[:,1]),
                                np.min(model_performance[:,1])]).reshape((1,4))
        
        summary = pd.DataFrame(summary_vals,
                               columns = ["N_LAYERS", "N_HIDDEN", "mean", "min"])
        
        model_summary = pd.concat((model_summary, summary))


# save the model summary
model_summary.to_csv(path / 'model_summary.csv', index=False)

