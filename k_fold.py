import pandas as pd
from sklearn.model_selection import KFold



kf=KFold(n_splits=10, shuffle=True, random_state=42)
data=pd.read_csv('cleaned_data/train.csv')
tree_data = pd.read_csv('cleaned_data/tree_train.csv')
'''for train_index, test_index in kf.split(data):
    data.filter(items=test_index, axis=0)
    print(data)'''