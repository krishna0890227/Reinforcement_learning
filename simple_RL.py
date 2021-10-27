

################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy import norm
# sample data
sample=np.random.normal(50, 9, 1000)
mean=np.mean(sample)
std=np.std(sample)
dist=norm(mean, std)
# expected outcomes and probabilities
values=[value for value in range (30, 70)]
probabilities= [dist.pdf(value) for value in values]
probabilities=np.exp(probabilities)
# histogram and pdf
plt.plot(values, probabilities)
plt.hist(sample, bins=10, density=True)
plt.show()





