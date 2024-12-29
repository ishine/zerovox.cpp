#!/bin/env python3

import numpy as np
import json

def instance_normalization(data, weight, bias, epsilon=1e-8):
    """
    Normalize the input data to have zero mean and unit variance for each instance.
    
    Parameters:
    - data: numpy array of shape (num_instances, num_features)
    - epsilon: small value to prevent division by zero
    
    Returns:
    - normalized_data: numpy array of the same shape as input data
    """
    
    # Calculate the mean for each instance (row)
    mean = np.mean(data, axis=1, keepdims=True)
    
    # Calculate the standard deviation for each instance (row)
    std = np.std(data, axis=1, keepdims=True)
    
    # Normalize the data
    normalized_data = (data - mean) / (std + epsilon)
    
    # apply weight and bias
    y = (normalized_data.transpose() * weight + bias).transpose()

    return y


with open ('utils/norm1dexample.json') as jsonf:
    data = json.load(jsonf)

x_in  = np.array(data['x_in'])
x_out = np.array(data['x_out'])

weight = np.array(data['weight'])
bias   = np.array(data['bias'])

normalized_data = instance_normalization(x_in[0], weight, bias)

print (f"normalized_data: {normalized_data}")



# dbg [528, 115  ne[0]=115 ne[1]=528 ] f32 = 
#   [
#     [-0.33291 -0.33291 -0.33291  ... -1.13479 -1.13479 -1.13479 ]
#     [-0.41624 -0.41624 -0.41624  ... -0.16569 -0.16569 -0.16569 ]
#     [-0.54738 -0.54738 -0.54738  ... -0.81904 -0.81904 -0.81904 ]
#     ...
#     [-0.17053 -0.17053 -0.17053  ... 1.81591 1.81591 1.81591 ]
#     [-0.80029 -0.80029 -0.80029  ... -0.59292 -0.59292 -0.59292 ]
#     [-0.14219 -0.14219 -0.14219  ... 0.18809 0.18809 0.18809 ]
#   ]
# sum:  -0.000063
