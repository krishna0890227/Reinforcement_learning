
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sess=tf.Session()


x=np.linspace(-20, 20, 41)
print(x)

y=tf.nn.relu(x)
y_values=sess.run(y)
plt.plot(x, y_values, 'r-*')
plt.xlabel('X_values')
plt.ylabel('y-values')
plt.show()




