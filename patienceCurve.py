import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Jeff Krichmar - UC Irvine
% 
% Plot probability to wait using a cumulative normal distribution.
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

mean = 70
std_dev = 20
BETA_WAIT = 2
delays = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130])
pWait = np.zeros([delays.shape[0]])

for i in range(delays.shape[0]):
    x = BETA_WAIT*norm.cdf(delays[i], loc=mean, scale=std_dev)
    pWait[i] = 1 / np.exp(x)

plt.plot(delays, pWait, '-o', label='Data Series 1')
plt.xlabel('Delay(s)')
plt.ylabel('pWait')
plt.show()
