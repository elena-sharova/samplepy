from rejection import Rejection
from importance import Importance
from metropolis_hastings import MH
import matplotlib.pyplot as plt
import numpy as np

#f= lambda x: 2.0*np.exp(-2.0*x)
#f= lambda x: 0.3*np.exp(-(x-0.3)**2) + 0.7* np.exp(-(x-2.)**2)
#f= lambda x: 0.3*np.exp(-(x-0.3)**2)* 0.7* np.exp(-(x-2.)**2)
#f= lambda x: 0.4*np.exp(-(x-0.3)**2/2.0)
#f = lambda x: np.exp(-1.0*x**2)*(2.0+np.sin(5.0*x)+np.sin(2.0*x))
#f = lambda x: 0.5*(x*np.exp(-1*x))/(1+x)

rej = Rejection(f, [0.01, 3.0])

imp = Importance(f, [0.01, 3.0])

mh = MH(f, [0.01, 3.0])


#sample = rej.sample(10000, 1)
#sample = imp.sample(10000, 0.03, 0.01)
sample = mh.sample(10000, 100, 200, 1)

x = np.arange(0.01, 3.0,(3.0-0.01)/10000)
fx = f(x)

figure, axis =plt.subplots()
axis.hist(sample, normed=1, bins=40)
axis2 = axis.twinx()
axis2.plot(x, fx, 'g')
plt.show()