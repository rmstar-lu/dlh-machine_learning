
# Optimization

## `0-norm_constants.py`

A function that calculates the normalization (standardization) constants of a matrix.

## `1-normalize.py`

A function that standardizes a matrix.

## `2-shuffle_data.py`

A function that shuffles the data points in two matrices the same way.

## `3-mini_batch.py`

A function that creates mini batches to be used for training a neural network using mini-batch gradient descent.

## `4-moving_average.py`

A function that calculates the weighted moving average of a data set using bias correction.

## `5-momentum.py`

A function that updates a variable using the gradient descent with momentum optimization algorithm:
```
    v = beta1 * v + (1 - beta1) * grad
    weight -= alpha * v
```

## `6-momentum.py`

A function that creates a momentum optimizer in tensorflow. Note the update rule used here is different to
the one in 5-momentum.py:
```
    v = beta1 * v - alpha * grad
    weight += v
```

## `7-RMSProp.py`

A function that updates a variable using the RMSProp optimization algorithm.

## `9-Adam.py`

A function that updates a variable using the Adam optimization algorithm.

## `11-learning_rate_decay.py`

A function that updates the learning rate using inverse time decay in numpy.

## `12-learning_rate_decay.py`

A function that creates an inverse time decay learning rate scheduler in tensorflow.
