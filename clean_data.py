import numpy as np
import pandas as pd


train_df = pd.read_csv('raw_data/train.csv')
test_df = pd.read_csv('raw_data/test.csv')
#print(train_df)
#print(test_df)

class CleanData:
    def __init__(self, df):
        self.df = df

    '''提取title'''
    def extract_title(self):
        self.df['Title'] = self.df['Name'].str.extract('([a-zA-Z]+)\.', expand = False)
        title_mapping={
            'Mlle': 'Miss',
            'Ms': 'Miss',
            'Mme': 'Mrs',
        }
        self.df['Title'] = self.df['Title'].map(title_mapping).fillna(self.df['Title'])
        # 2. 合并低频Title
        self.df['Title'] = self.df['Title'].replace(['Capt', 'Rev', 'Dr', 'Col', 'Major'], 'Officer')
        self.df['Title'] = self.df['Title'].replace(['Jonkheer', 'Don', 'Dona', 'Countess', 'Lady', 'Sir'], 'Royal')
        print(self.df['Title'].value_counts())
        return self

    '''填充age'''
    def imputed_age(self):
        self.df['Age_imputed'] = self.df['Age'].isnull().astype(int)
        imputed_data = self.df.groupby(['Title', 'Pclass'])['Age'].agg('median')
        self.df['Age'] = self.df.groupby(['Title', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
        print(imputed_data)
        return self

    '''清除多余数据'''
    def clean(self):
        self.df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
        return self

    '''填充embarked'''
    def fill_Embarked(self):
        self.df['Embarked'].fillna('S')
        return self

    '''one hot encode'''
    def one_hot_encode(self,names):
        for name in names:
            encoded=pd.get_dummies(self.df[name],prefix=name).astype(int)
            self.df=pd.concat([self.df.drop([name],axis=1),encoded],axis=1)
        return self
    '''encode sex: female->0, male->1'''
    def encode(self):
        self.df.replace({'male':1,'female':0},inplace = True)
        self.df = self.df.rename(columns={'Sex': 'Is_male'})
        return self

    '''log fare'''
    def log_feature(self, feature):
        print(self.df.columns)
        self.df[feature] = np.log(self.df[feature] + 1)
        return self

    '''归一化'''
    def normalization(self):
        label = None
        if 'Survived' in self.df.columns:
            label = self.df['Survived']
            self.df.drop(columns=['Survived'], inplace = True)
        df_mean = self.df.mean()
        df_std = self.df.std()
        self.df = (self.df - df_mean) / (df_std + 1e-8)     # 特征标准化
        if label is not None:
            self.df = pd.concat([label, self.df],axis=1)
        return self

    def save(self, path):
        self.df.to_csv(path, index=False)

    #决策树数据处理
    '''Embark编码'''
    def tree_encode_map(self):
        embarked_mapping = self.df.groupby('Embarked')['Survived'].mean().to_dict()
        title_mapping = self.df.groupby('Title')['Survived'].mean().to_dict()
        mapping = {
            'Embarked' : embarked_mapping,
            'Title' : title_mapping
        }
        return mapping
    def tree_encode(self, names, mapping):
        for name in names:
            self.df[name] = self.df[name].map(mapping[name])
        return self

    #print
    def __str__(self):
        return str(self.df)

train_data = (CleanData(train_df).extract_title().imputed_age().clean().fill_Embarked()
              .one_hot_encode(['Title','Embarked']).encode().log_feature('Fare').normalization())
train_df = train_data.df
train_data.save('cleaned_data/train.csv')

test_data = (CleanData(test_df).extract_title().imputed_age().clean().fill_Embarked()
             .one_hot_encode(['Title','Embarked']).encode().log_feature('Fare').normalization())
test_df = test_data.df
test_data.save('cleaned_data/test.csv')
print(train_df)
print(test_df)


'''tree_train_data = CleanData(train_df).extract_title().clean()
mapping = tree_train_data.tree_encode_map()
tree_train_data.tree_encode(['Title','Embarked'], mapping).encode()
tree_train_data.save('cleaned_data/tree_train.csv')
print(tree_train_data)

tree_test_data = CleanData(test_df).extract_title().clean().tree_encode(['Title','Embarked'], mapping).encode()
tree_test_data.save('cleaned_data/tree_test.csv')
print(tree_test_data)'''

