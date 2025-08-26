import os

import numpy as np
import pandas as pd
from pandas import DataFrame

from scipy.stats import randint
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted, check_array
from tqdm import tqdm




def bootstrap_sample(X, y):
    n_samples = len(y)
    indices = np.random.choice(n_samples, n_samples, replace=True)
    X_bootstrap = X[indices]
    y_bootstrap = y[indices]
    return X_bootstrap, y_bootstrap


class MyRandomForest(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 n_estimators = 100,
                 n_jobs = 1,
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 max_features = 'auto',
                 bootstrap=True,
                 random_state=None,
                 random_hargs = False
                 ):
        self.classes_ = None
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.random_hargs = random_hargs
        self.n_features_ = None
        self.estimators_ = []

    def fit(self, X, y):
        #传入df
        if isinstance(X, DataFrame):
            X = np.array(X)
            y = np.array(y).reshape(-1)
        # 输入验证
        X, y = check_X_y(X, y)

        m, n =np.shape(X)
        self.n_features_ = n
        self.classes_ = y

        # 确定每个节点考虑的特征数量
        if self.max_features == 'auto' or self.max_features == 'sqrt':
            max_features = int(np.sqrt(self.n_features_))
        elif self.max_features == 'log2':
            max_features = int(np.log2(self.n_features_))
        elif isinstance(self.max_features, int):
            max_features = self.max_features
        elif isinstance(self.max_features, float):
            max_features = int(self.max_features * self.n_features_)
        else:
            max_features = self.n_features_

        for i in tqdm(range(self.n_estimators), total=self.n_estimators):
            estimator = self._train_tree(X, y, i)
            self.estimators_.append(estimator)
        return self

    def _train_tree(self, X, y, i):
        X_bootstrap, y_bootstrap = bootstrap_sample(X, y)
        estimator = DecisionTreeClassifier(
            criterion='gini',
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.n_features_,
            random_state=self.random_state + i if self.random_state is not None else None
        )

        estimator.fit(X_bootstrap, y_bootstrap)
        return estimator

    def predict_proba(self, X):
        # 检查是否已经拟合
        check_is_fitted(self, 'estimators_')
        X = check_array(X)

        # 收集所有树的概率预测
        probas = [tree.predict_proba(X) for tree in self.estimators_]

        # 平均概率
        avg_proba = np.mean(probas, axis=0)
        return avg_proba

    def predict(self, X):
        # 使用 predict_proba 实现 predict
        check_is_fitted(self, 'estimators_')
        X = check_array(X)

        # 获取概率预测
        proba = self.predict_proba(X)

        # 选择概率最高的类别
        return self.classes_[np.argmax(proba, axis=1)]

if __name__ == "__main__":
    data = pd.read_csv('cleaned_data/rf_train.csv')
    y = data['Survived']
    X = data.drop(columns=['Survived'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    rf_estimator = MyRandomForest()
    # 定义参数分布
    param_dist = {
        'n_estimators': randint(500, 2000),
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['auto', 'log2']
    }

    # 创建随机搜索实例（使用内部并行）
    random_search = RandomizedSearchCV(
        rf_estimator,
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1
    )
    # 执行搜索
    random_search.fit(X, y)

    print("最佳参数:", random_search.best_params_)
    print("最佳得分:", random_search.best_score_)

    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import classification_report

    # 1. Dummy 基线
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(X_train, y_train)
    print('Dummy acc:', dummy.score(X_test, y_test))

    # 2. 类别不平衡
    print(y_train.value_counts(normalize=True))

    # 3. 重跑轻量随机森林
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=42
    )
    rf.fit(X_train, y_train)
    print('RF acc:', rf.score(X_test, y_test))
    print(classification_report(y_test, rf.predict(X_test)))

    my_estimator = MyRandomForest(
        max_depth=random_search.best_params_['max_depth'],
        min_samples_split=random_search.best_params_['min_samples_split'],
        max_features=random_search.best_params_['max_features'],
        n_estimators=random_search.best_params_['n_estimators'],
        min_samples_leaf=random_search.best_params_['min_samples_leaf']
    )
    my_estimator.fit(X, y)
    test_data = pd.read_csv('cleaned_data/rf_test.csv')
    raw_test_data = pd.read_csv('raw_data/test.csv')
    y_pred = my_estimator.predict(test_data)
    submission = pd.DataFrame({
        'PassengerId': raw_test_data['PassengerId'],
        'Survived': y_pred.astype(int)
    })
    print(my_estimator.predict(test_data))
    submission.to_csv('rf_result.csv', index=False)