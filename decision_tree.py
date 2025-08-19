import pandas as pd
import numpy as np


from k_fold import *

#分裂基准   ( : ]
'''信息熵'''
def Ent(df):
    p = df['Survived'].mean()
    q = 1 - p
    return -p*np.log2(p) - q*np.log2(q)
'''信息增益'''
def gain(df, name, boundary):
    ent1 = Ent(df.loc[df[name] > boundary, 'Survived'])
    ent2 = Ent(df.loc[df[name] <= boundary, 'Survived'])
    ent = Ent(df)
    p1 = df.loc[df[name] > boundary, 'Survived'].size / df['Survived'].size
    p2 = 1 - p1
    return ent - p1 * ent1 - p2 * ent2
'''信息增益率'''
def gain_rate(df, name, boundary):
    p = df.loc[df[name] > boundary, 'Survived'].size / df['Survived'].size
    q = 1 - p
    tv = -p*np.log2(p) - q*np.log2(q)
    return gain(df, name, boundary)/tv
'''Gini'''
def gini(df):
    p = df['Survived'].mean()
    q = 1 - p
    return 1 - p**2 - q**2
'''Gini指数, 实际返回1 - gini_index'''
def neg_gini_index(df, name, boundary):
    p1 = df.loc[df[name] > boundary, 'Survived'].size / df['Survived'].size
    p2 = 1 - p1
    gini_ = p1*gini(df[df[name] > boundary]) + p2*gini(df[df[name] <= boundary])
    return 1 - gini_

#选择最佳分类点
def choose_boundary(data, name, f):
    best_boundary = 0
    best_gain = 0
    #连续值
    q = 10
    if data[name].nunique(dropna=True) > q:
        bins = pd.qcut(data[name], q, retbins= True, duplicates = 'drop')[1]
        for i in range(1, np.size(bins) - 2):
            if f(data, name, bins[i]) > best_gain:
                best_boundary = bins[i]
                best_gain = f(data, name, best_boundary)
    #0 or 1
    elif data[name].nunique(dropna=True) == 2:
        best_boundary = 0.5
        best_gain = f(data, name, 0.5)
    #离散值
    else:
        val_list = np.unique(data[name])
        boundary_list = (val_list[1:]+val_list[:-1])/2
        best_gain = 0
        for b in boundary_list:
            if f(data, name, b) > best_gain:
                best_gain = f(data, name, b)
                best_boundary = b
    return best_boundary, best_gain

#选择最佳分类特征
def choose_feature(data, f):
    best_gain = 0
    best_feature = ''
    best_boundary = 0
    for name in data.columns:
        if name != 'Survived':
            boundary, gain = choose_boundary(data, name, f)
            if gain > best_gain:
                best_gain = gain
                best_feature = name
                best_boundary = boundary
    return best_feature, best_boundary

#树分裂
'''分裂后处理'''
def possess(data, nan_data, name, f):
    data_ = pd.concat([data, nan_data])
    tree_dict = {}
    if data[name].nunique(dropna=True) > 1 and data_['Survived'].nunique(dropna=True) > 1:
        tree_dict = build_tree(data_, f)
    elif data[name].nunique(dropna=True) == 1 and data_['Survived'].nunique(dropna=True) > 1:
        data.drop([name], axis=1)
        if data.columns.size == 1:
            w_sum = data_['weight'].sum()
            prob = np.sum(np.array(data_['weight']) * np.array(data_['Survived'])) / w_sum
            tree_dict = prob
    else:
        tree_dict = data['Survived'].iloc[0]
    return tree_dict
'''{name > boundary: {0: {}, 1: {}}'''
def build_tree(data, f):
    name, boundary = choose_feature(data, f)
    print(name)
    data1 = data[data[name] <= boundary]
    data2 = data[data[name] > boundary]
    data3 = data[data[name].isna()]

    #calculate weight
    true_sum = data1['weight'].sum()
    false_sum = data2['weight'].sum()
    #weight = true rate
    weight = true_sum/(true_sum + false_sum)
    data3_t = data3.copy()
    data3_f = data3.copy()
    data3_t['weight'] = data3_t['weight'] * weight
    data3_f['weight'] = data3_f['weight'] * (1 - weight)

    false_dict = possess(data1, data3_f, name, f)
    true_dict = possess(data2, data3_t, name, f)
    return {
        'feature': name,
        'boundary': boundary,
        'False': false_dict,
        'True': true_dict,
        'weight':weight
    }
#分类
'''单个分类'''
def tree_classify(sample, tree):
    if isinstance(tree, dict):
        feature = tree['feature']
        boundary = tree['boundary']
        weight = tree['weight']
        if pd.isna(sample[feature]):
            res = weight * tree_classify(sample, tree['True']) + (1 - weight) * tree_classify(sample, tree['False'])
        else:
            res = tree_classify(sample, tree[str(sample[feature] > boundary)])
    else:
        res = tree
    return res
'''全体分类'''
def data_tree_classify(data, tree):
    return int(tree_classify(data, tree) > 0.5)

if __name__ == '__main__':
    err_list = []
    for train_index, test_index in kf.split(tree_data):
        #加权重
        train_data = tree_data.filter(items=train_index, axis=0)
        train_data['weight'] = 1
        test_data = tree_data.filter(items=test_index, axis=0)
        test_data['weight'] = 1

        #构造树
        tree = build_tree(train_data, neg_gini_index)
        true_survived = test_data['Survived']
        res = data_tree_classify(test_data, tree)
        error = (res - true_survived).mean()
        err_list.append(error)
    print(err_list)

