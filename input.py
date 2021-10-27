
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

RawNet_data=pd.read_excel("data/LD_NET.xlsx")
print(RawNet_data.head())
Net_data=RawNet_data.iloc[6000:10000, 1:]

print(Net_data.head())

NetWork_data=Net_data.values
Network_data=NetWork_data.flatten()

plt.plot(Network_data)
plt.show()



