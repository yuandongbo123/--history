import bisect
from bisect import bisect_right
import matplotlib.pyplot as plt
warmup_factor = 0.001
Steps = (300,400)
gamma = 0.1
warmup_iters = 1000
base_lr = 0.001
import numpy as np
lr = []
iters=[]
for iter in range(500):
    alpha = iter/warmup_iters
    warmup_factor = warmup_factor*(1-alpha)+alpha
    lr.append( base_lr * warmup_factor * gamma ** bisect_right(Steps, iter))
    iters.append(iter)
plt.plot(iters,lr)
plt.show()
