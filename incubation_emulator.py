# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 10:03:06 2023

@author: Jonathan.Carruthers
"""

from tensorflow import keras
import mdn
import numpy as np
from pathlib import Path
import pandas as pd

import mdn_funcs

''' Perform a grid search over hyperparameters for a given architecture '''

# load the training data
sub = []
for i in range(5):
    sub.append(np.load(f'training_data/incubation_periods_{i}.npy'))

train_data = np.concatenate(sub)
X_train = train_data[:,:-1]
Y_train = np.log(train_data[:,-1])
print(f'Using {X_train.shape[0]} training sets')


# load the test data, the first four columns contain model parameters, the
# remaining 1000 columns contain sample incubation periods
test_data = np.load("test_data/incubation_test.npy")
n_test_sets = test_data.shape[0]
X_test = test_data[:,:4]
Y_test = np.log(test_data[:,4:])

# specify the architecture of the MDN
N_LAYERS = 3
N_HIDDEN = 64
N_MIXES = 10

# lower and upper bounds for L2 regularisation and learning rates (log10 scale)
l2_reg_low, l2_reg_high = -3, 1.7
lr_low, lr_high = -3.3, -2.5

# path for saving models
path = Path(f'incubation_models/NLAYERS={N_LAYERS}_NHIDDEN={N_HIDDEN}_NMIXES={N_MIXES}')
path.mkdir(parents=True, exist_ok=True)

# specify the number of models to try
n_models = 1
model_summary = pd.DataFrame()

for nm in range(n_models):
    model_attributes = {'model number': nm+1,
                        'N_LAYERS': N_LAYERS,
                        'N_HIDDEN': N_HIDDEN,
                        'N_MIXES': N_MIXES}
    
    # Sample the L2 regularization parameter
    l2_reg = 10**np.random.uniform(l2_reg_low, l2_reg_high)
    learning_rate = 10**np.random.uniform(lr_low, lr_high)
    
    # Define the model
    model = keras.Sequential()
    model.add(keras.layers.Dense(N_HIDDEN, batch_input_shape=(None, 4), activation='relu',
              kernel_regularizer=keras.regularizers.l2(l2=l2_reg)))
    for _ in range(N_LAYERS-1):
        model.add(keras.layers.Dense(N_HIDDEN, activation='relu',
                  kernel_regularizer=keras.regularizers.l2(l2=l2_reg)))
    model.add(mdn.MDN(1, N_MIXES))

    model.compile(loss=mdn.get_mixture_loss_func(1, N_MIXES), 
                  optimizer=keras.optimizers.Adam(learning_rate=learning_rate))

    # Sample the hyperparameters
    batch_size = np.random.choice([1024, 2048, 4096])
    epochs = 250
    model_attributes["batch_size"] = batch_size
    model_attributes["epochs"] = epochs
    model_attributes["learning_rate"] = np.round(learning_rate, 6)
    model_attributes["L2_reg"] = np.round(l2_reg, 6)
    
    # Train the model
    history = model.fit(x=X_train, y=Y_train, batch_size=batch_size, epochs=epochs, verbose=2)
    
    # Test the model using the test set
    Y_pred = model.predict(X_test)
    
    # Pull out the means, standard deviations and weights from the output
    mus = Y_pred[:,:N_MIXES]
    sigs = Y_pred[:,N_MIXES:2*N_MIXES]
    pis = mdn_funcs.softmax(Y_pred[:,2*N_MIXES:])
    
    # Using the scaled KS test statistic
    KS, n_approx_i95 = mdn_funcs.scaled_KS_score(Y_test, pis, mus, sigs)
    model_attributes["n_approx_i95"] = n_approx_i95
    
    # Metrics to evaluate the performance of a model
    metrics = [(KS > z).sum()/n_test_sets for z in [0.025, 0.05, 0.1]]
    model_attributes["KS>0.025"] = np.round(metrics[0], 4)
    model_attributes["KS>0.05"] = np.round(metrics[1], 4)
    model_attributes["KS>0.1"] = np.round(metrics[2], 4)

    model_attributes["final_loss"] = np.round(history.history['loss'][-1], 4)
    
    # Store the hyperparameters and metrics in a dataframe
    model_summary = pd.concat((model_summary, pd.DataFrame(model_attributes, index=[nm])))
    
    # Save the model
    model.save(path / f'model_{nm+1}.h5')
        
    # Save the dataframe of hyperparameters and metrics after each model fit
    model_summary.to_csv(path / 'model_summary.csv', index=False)
