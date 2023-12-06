# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 13:23:11 2023

@author: Jonathan.Carruthers
"""

import numpy as np

# splitting up the training data
X = np.load('training_data/incubation_periods.npy')

for i in range(5):
    sub = X[i*1_000_000 : (i+1)*1_000_000]
    np.save(f'training_data/incubation_periods_{i}.npy', sub)


# reconstructing the training data
sub = []
for i in range(5):
    sub.append(np.load(f'training_data/incubation_periods_{i}.npy'))
       
Y = np.concatenate(sub)
print((Y==X).all())
