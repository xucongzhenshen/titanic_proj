from k_fold import *
import numpy as np
from matplotlib import pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

#预处理函数
def pros_data(data, index):
    train_data = data.filter(items=train_index, axis=0)
    y = train_data['Survived'].to_numpy()
    y = y.reshape(-1, 1)
    #X = train_data.drop(['Survived', 'Title_Royal'], axis=1).to_numpy()
    X=train_data.drop(['Survived', 'Fare'], axis=1).to_numpy()
    # log Fare
    fares_=np.log(train_data['Fare'] + 1).to_numpy().reshape(-1, 1)
    X = np.c_[X, fares_]

    # 添加偏置项（截距）
    X = np.c_[np.ones((X.shape[0], 1)), X]
    # 特征标准化
    X_mean = np.mean(X[:, 1:], axis=0)
    X_std = np.std(X[:, 1:], axis=0)
    X[:, 1:] = (X[:, 1:] - X_mean) / (X_std + 1e-8)
    return X,y

#off policy gradient ascend
def all_gd_asc(data, train_index, alpha, maxCycles, reg_lambda, improve_rate):
    X,y=pros_data(data,train_index)
    m, n = np.shape(X)

    '''loss history'''
    loss_history=[]

    weights = np.zeros((n, 1))
    best_loss = -1
    best_weights=[]
    patience = 10
    no_improve = 0
    improve_his = []

    for i in range(maxCycles):  # heavy on matrix operations
        z=X @ weights
        h = sigmoid(z)  # matrix mult
        error = y - h  # vector subtraction
        weights = weights + alpha * (X.T @ error) - reg_lambda * weights # matrix mult

        loss=np.mean((y-1)* z + np.log(h))
        loss_history.append(loss)
        improve_his.append(loss-best_loss)

        if loss - best_loss > improve_rate:
            best_loss = loss
            no_improve = 0
            best_weights = weights.copy()
        else:
            no_improve += 1

        # 每1000次迭代打印损失
        if i % 500 == 0:
            print(f"Iteration {i}: Loss = {loss:.4f}")
        #早停机制
        if no_improve >= patience:
            print(f"Early stopping at iteration {i}")
            weights = best_weights
            break
    #plot loss history
    '''plt.style.use('seaborn-v0_8-dark')
    fig, ax = plt.subplots()
    ax.plot(loss_history)
    plt.show()'''
    #print(improve_his)
    return weights

#分类函数
def classify(dataMat,weights):
    # 预测概率
    proba = sigmoid(dataMat @ weights)
    # 分类决策（阈值0.5）
    y_pred = (proba > 0.5).astype(int)
    return y_pred  # 返回决策

#测试函数
def test(data, test_index, weights):
    X, y=pros_data(data,test_index)
    y_pred=classify(X,weights)
    # 计算准确率
    accuracy = np.mean(y_pred == y)
    return 1 - accuracy  # 返回错误率

'''h = sigmoid(X @ weights)
error = y - h
count = 0
n,_ = np.shape(error)
for res in error:
    if abs(res[0]) > 1 / 2:
        count += 1
    elif abs(res[0]) == 1 / 2:
        count += 1 / 2
return count / n'''



if __name__=='__main__':
    ave_err=[]
    least_err_vec=[]
    least_err=1
    best_para=0
    '''
    合理参数：
    alpha   max     reg     impro       err
    0.0033  5000    0       2e-6        ?
    0.0027  5000    0.001   2e-6        0.16560
    不合理参数：

    '''
    for alpha in [i*0.0001 + 0.001 for i in range(1, 20)]:
        error = []
        for train_index, test_index in kf.split(data):
            w=all_gd_asc(data, train_index, alpha=alpha, maxCycles=5000, reg_lambda=0.001, improve_rate = 2e-6)
            error.append(test(data,test_index,w))
        ave_err.append(np.average(error))
        if ave_err[-1]<least_err:
            least_err=ave_err[-1]
            best_para=alpha
            least_err_vec=error
    print(ave_err)
    print(f'the best para is {best_para}, the least error is {least_err}')
    print(f'error vec is {least_err_vec}')
