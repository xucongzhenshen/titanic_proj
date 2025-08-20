import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

data = pd.read_csv('cleaned_data/train.csv')
x_val = data['Parch']
y_val = data['SibSp']
x_val = [x + np.random.normal(0, 0.1) for x in x_val]
y_val = [y + np.random.normal(0, 0.1) for y in y_val]
c_val = data['Survived']

plt.style.use('seaborn-v0_8-dark')
fig, ax = plt.subplots()
ax.scatter(x_val, y_val, c=c_val, cmap=plt.cm.Greens, s=2)
# ax.axis([-10,100,0,20])
plt.show()
