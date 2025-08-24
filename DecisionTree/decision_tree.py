import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score

from k_fold import *


# 信息熵
def Ent(df):
    p = df['Survived'].mean()
    q = 1 - p
    return -p * np.log2(p) - q * np.log2(q)
# 信息增益
def gain(df, name, boundary):
    ent1 = Ent(df.loc[df[name] > boundary, 'Survived'])
    ent2 = Ent(df.loc[df[name] <= boundary, 'Survived'])
    ent = Ent(df)
    p1 = df.loc[df[name] > boundary, 'Survived'].size / df['Survived'].size
    p2 = 1 - p1
    return ent - p1 * ent1 - p2 * ent2
# 信息增益率
def gain_rate(df, name, boundary):
    p = df.loc[df[name] > boundary, 'Survived'].size / df['Survived'].size
    q = 1 - p
    tv = -p * np.log2(p) - q * np.log2(q)
    return gain(df, name, boundary) / tv
# Gini
def gini(df):
    p = df['Survived'].mean()
    q = 1 - p
    return 1 - p ** 2 - q ** 2
# Gini指数, 实际返回1 - gini_index
def neg_gini_index(df, name, boundary):
    p1 = df.loc[df[name] > boundary, 'Survived'].size / df['Survived'].size
    p2 = 1 - p1
    gini_ = p1 * gini(df[df[name] > boundary]) + p2 * gini(df[df[name] <= boundary])
    return 1 - gini_

'''分裂基准   ( : ]'''
# 选择最佳分类点
def choose_boundary(data, name, f):
    best_boundary = 0
    best_gain = 0
    # 连续值
    q = 10
    if data[name].nunique(dropna=True) > q:
        bins = pd.qcut(data[name], q, retbins=True, duplicates='drop')[1]
        for i in range(1, np.size(bins) - 2):
            if f(data, name, bins[i]) > best_gain:
                best_boundary = bins[i]
                best_gain = f(data, name, best_boundary)
    # 0 or 1
    elif data[name].nunique(dropna=True) == 2:
        best_boundary = 0.5
        best_gain = f(data, name, 0.5)
    # 离散值
    else:
        val_list = np.unique(data[name])
        boundary_list = (val_list[1:] + val_list[:-1]) / 2
        best_gain = 0
        for b in boundary_list:
            if f(data, name, b) > best_gain:
                best_gain = f(data, name, b)
                best_boundary = b
    return best_boundary, best_gain
# 选择最佳分类特征
def choose_feature(data, f):
    best_gain = 0
    best_feature = None
    best_boundary = 0
    for name in data.columns:
        if name not in ['Survived', 'weight']:
            boundary, gain = choose_boundary(data, name, f)
            if gain > best_gain:
                best_gain = gain
                best_feature = name
                best_boundary = boundary
    return best_feature, best_boundary, best_gain

'''树分裂'''
# 分裂后处理
def possess(data, nan_data, name, f, hargs, depth):
    data_ = pd.concat([data, nan_data])
    if data[name].nunique(dropna=True) == 1:
        data_.drop([name], axis=1)
    if data_['Survived'].nunique(dropna=True) == 1 or data.columns.size == 1 or len(data) < 5:
        tree_dict = cal_label(data_)
    else:
        tree_dict = build_tree(data_, f, hargs=hargs, depth = depth + 1)
    return tree_dict
# {name > boundary: {0: {}, 1: {}}, weight = true rate
def build_tree(data, f, hargs, depth=0):
    name, boundary, gain = choose_feature(data, f)
    # 提前终止条件
    n_samples = len(data)
    total_weight = data['weight'].sum()

    # 更多终止条件
    max_depth, min_samples_split, min_impurity, alpha, min_samples_leaf = hargs
    stop_conditions = [
        depth >= max_depth,  # 最大深度
        n_samples < min_samples_leaf,  # 最小样本数
        total_weight < min_samples_split,  # 最小权重和
        data['Survived'].nunique() == 1,  # 纯节点
        cal_impurity(data) < min_impurity  # 杂质足够低
    ]

    if any(stop_conditions):
        return cal_label(data)
    else:
        data1 = data[data[name] <= boundary]
        data2 = data[data[name] > boundary]
        data3 = data[data[name].isna()]

        # calculate weight
        true_sum = data1['weight'].sum()
        false_sum = data2['weight'].sum()
        # weight = true rate
        weight = true_sum / (true_sum + false_sum + 1e-10)
        data3_t = data3.copy()
        data3_f = data3.copy()
        data3_t['weight'] = data3_t['weight'] * weight
        data3_f['weight'] = data3_f['weight'] * (1 - weight)

        false_dict = possess(data1, data3_f, name, f, hargs=hargs, depth=depth)
        true_dict = possess(data2, data3_t, name, f, hargs=hargs, depth=depth)
        branch = {
            'feature': name,
            'boundary': boundary,
            'False': false_dict,
            'True': true_dict,
            'weight': weight
        }
    return branch
# 计算标签
def cal_label(data):
    w_sum = data['weight'].sum()+1e-10
    prob = np.sum(np.array(data['weight']) * np.array(data['Survived'])) / w_sum
    return prob
# 计算杂度
def cal_impurity(data):
    prob = cal_label(data)
    return min(prob, 1 - prob)

'''分类'''
# 单个分类
def tree_classify(sample, tree):
    if not is_leaf(tree):
        feature = tree['feature']
        boundary = tree['boundary']
        weight = tree['weight']  # weight = true rate
        if pd.isna(sample[feature]):
            res = weight * tree_classify(sample, tree['True']) + (1 - weight) * tree_classify(sample, tree['False'])
        else:
            res = tree_classify(sample, tree[str(sample[feature] > boundary)])
    else:
        res = tree
    return res
# 线程池
num_workers = os.cpu_count()
GLOBAL_EXECUTOR = ThreadPoolExecutor(max_workers=num_workers)
# 全体分类
def data_tree_classify(data, tree):
    def process_data(data):
        return [int(tree_classify(row, tree) > 0.5) for _, row in data.iterrows()]

    future = GLOBAL_EXECUTOR.submit(process_data, data)
    return future.result()
# 剪枝
def is_leaf(tree):
    return not isinstance(tree, dict)
def cut_branch(data, tree, alpha):
    name = tree['feature']
    boundary = tree['boundary']
    data1 = data[data[name] <= boundary]
    data2 = data[data[name] > boundary]
    data3 = data[data[name].isna()]

    weight = tree['weight']
    data3_t = data3.copy()
    data3_f = data3.copy()
    data3_t['weight'] = data3_t['weight'] * weight
    data3_f['weight'] = data3_f['weight'] * (1 - weight)
    data_t = pd.concat([data2, data3_t])
    data_f = pd.concat([data1, data3_f])
    if not is_leaf(tree['True']):
        tree['True'] = cut_branch(data_t, tree['True'], alpha)
    if not is_leaf(tree['False']):
        tree['False'] = cut_branch(data_f, tree['False'], alpha)
    if is_leaf(tree['True']) and is_leaf(tree['False']):
        impurity0 = cal_impurity(data)
        impurity_t = cal_impurity(data_t)
        impurity_f = cal_impurity(data_f)
        if impurity0 + alpha<= (1 - weight) * impurity_f + weight * impurity_t:
            tree = cal_label(data)
    return tree


# 画树
def plt_tree(tree, depth=0):
    null_str = ""
    for i in range(depth):
        null_str += '\t'
    print(f'{null_str}\033[91m{tree["feature"]} > {tree["boundary"]}\033[0m, weight: {tree["weight"]}')
    tree_t = tree['True']
    tree_f = tree['False']
    if is_leaf(tree_t):
        print(f'{null_str}True: class: {tree_t}')
    else:
        print(f'{null_str}True: {{')
        plt_tree(tree_t, depth + 1)
        print(f'{null_str}}}')
    if is_leaf(tree_f):
        print(f'{null_str}False: class: {tree_f}')
    else:
        print(f'{null_str}False: {{')
        plt_tree(tree_f, depth + 1)
        print(f'{null_str}}}')


# 评估性能
def train_and_evaluate_fold(args):
    train_index, test_index, hargs = args
    max_depth, min_samples_split, min_impurity, alpha , min_samples_leaf= hargs
    # 加权重
    train_data = tree_data.filter(items=train_index, axis=0)
    train_data['weight'] = 1
    test_data = tree_data.filter(items=test_index, axis=0)
    test_data['weight'] = 1

    # 构造树
    tree = build_tree(train_data, neg_gini_index, hargs=hargs)
    true_survived = test_data['Survived']
    # 剪枝前
    '''former_res = data_tree_classify(test_data, tree)
    former_error = np.mean(np.array(former_res) != np.array(true_survived))'''
    # 剪枝后
    tree = cut_branch(test_data, tree, alpha)
    #print(tree)
    res = data_tree_classify(test_data, tree)
    error = np.mean(np.array(res) != np.array(true_survived))
    if error<0.8:
        plt_tree(tree)
    return error

if __name__ == '__main__':
    min_samples_split = 5
    min_samples_leaf = 4
    min_impurity = 0
    alpha = 0
    max_depth = 7
    max_workers = None
    hargs=max_depth, min_samples_split, min_impurity, alpha, min_samples_leaf
    # 准备参数
    args_list = [(train_index, test_index, hargs)
                 for train_index, test_index in kf.split(tree_data)]
    # 使用进程池并行处理
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        err_list = list(executor.map(train_and_evaluate_fold, args_list))


    print(err_list)

'''[np.float64(0.13333333333333333), 
np.float64(0.1348314606741573), 
np.float64(0.14606741573033707), 
np.float64(0.1797752808988764), 
np.float64(0.11235955056179775), 
np.float64(0.07865168539325842), 
np.float64(0.19101123595505617), 
np.float64(0.21348314606741572), 
np.float64(0.19101123595505617), 
np.float64(0.07865168539325842)]'''

'''[np.float64(0.13333333333333333), 
np.float64(0.1348314606741573), 
np.float64(0.11235955056179775), 
np.float64(0.16853932584269662), 
np.float64(0.12359550561797752), 
np.float64(0.06741573033707865), 
np.float64(0.19101123595505617), 
np.float64(0.21348314606741572), 
np.float64(0.2247191011235955), 
np.float64(0.056179775280898875)]'''