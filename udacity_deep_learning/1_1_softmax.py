"""Softmax."""

scores = [1.0, 2.0, 3.0]

import numpy as np

def softmax_x(x, den):
    return np.exp(x)/den

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    array = np.array(x, dtype='float64')
    vsoftmax = np.vectorize(softmax_x)
    if(array.ndim == 1):
        # 1d array
        denominateur = sum(np.exp(x))
        array = vsoftmax(array, denominateur)
    else:
        # 2d array
        for col_id in range(array.shape[1]):
            denominateur = sum(np.exp(array[:,col_id]))
            array[:,col_id] = vsoftmax(x[:,col_id], denominateur)
    return array

print(softmax(scores))

scores2 =  np.array([[1., 2, 3, 6],
                   [2, 4, 5, 6],
                   [3, 8, 7, 6]])
print(softmax(scores2))

# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()
