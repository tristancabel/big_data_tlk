import numpy as np
import matplotlib.pyplot as plt

greyhounds = 500
labs = 500

grey_height = 28 + 4* np.random.rand(greyhounds)
lab_height = 24 + 4* np.random.rand(labs)

plt.hist([grey_height, lab_height], sstacked=True, color['r','b'])
plt.show()
