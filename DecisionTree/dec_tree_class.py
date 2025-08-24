import logging
import multiprocessing
import time
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import make_classification
from tqdm import tqdm
import threading
from collections import defaultdict

# 全局进度跟踪器
class ProgressTracker:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ProgressTracker, cls).__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        self.total_fits = 0
        self.completed_fits = 0
        self.start_time = None
        self.n_jobs = 0
        self.fit_times = defaultdict(list)

    def init_progress(self, total_fits, n_jobs):
        self.total_fits = total_fits
        self.completed_fits = 0
        self.start_time = time.time()
        self.n_jobs = n_jobs

    def update_progress(self, params, fit_time):
        self.completed_fits += 1
        # 记录每种参数组合的拟合时间
        param_key = str({k: v for k, v in params.items() if k != 'verbose'})
        self.fit_times[param_key].append(fit_time)


        # 计算估计剩余时间
        elapsed = time.time() - self.start_time
        avg_time_per_fit = elapsed / self.completed_fits
        remaining_fits = self.total_fits - self.completed_fits
        remaining_time = avg_time_per_fit * remaining_fits

        print(f"已训练: {self.completed_fits}fits \
                已用时间: {elapsed:.1f}s \
                剩余时间: {remaining_time:.1f}s \
                平均时间/拟合: {avg_time_per_fit:.2f}s")

    def finish(self):
        # 打印摘要信息
        elapsed = time.time() - self.start_time
        print(f"\n训练完成! 总共 {self.total_fits} 次拟合，耗时 {elapsed:.1f} 秒")
        print(f"平均每次拟合时间: {elapsed / self.total_fits:.2f} 秒")

        # 找出最快和最慢的参数组合
        if self.fit_times:
            avg_times = {k: sum(v) / len(v) for k, v in self.fit_times.items()}
            fastest = min(avg_times.items(), key=lambda x: x[1])
            slowest = max(avg_times.items(), key=lambda x: x[1])

            print(f"最快的参数组合: {fastest[0]} (平均 {fastest[1]:.2f} 秒/拟合)")
            print(f"最慢的参数组合: {slowest[0]} (平均 {slowest[1]:.2f} 秒/拟合)")



# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)

# 信息熵
def Ent(series):
    p = series.mean()
    if p == 0 or p == 1:
        return 0
    q = 1 - p
    return -p * np.log2(p) - q * np.log2(q)


# 信息增益
def gain(df, name, boundary, target_col='target'):
    ent1 = Ent(df.loc[df[name] > boundary, target_col])
    ent2 = Ent(df.loc[df[name] <= boundary, target_col])
    ent = Ent(df[target_col])
    p1 = df.loc[df[name] > boundary, target_col].size / df[target_col].size
    p2 = 1 - p1
    return ent - p1 * ent1 - p2 * ent2


# 信息增益率
def gain_rate(df, name, boundary, target_col='target'):
    p = df.loc[df[name] > boundary, target_col].size / df[target_col].size
    q = 1 - p
    if p == 0 or q == 0:
        return 0  # 避免除以零
    tv = -p * np.log2(p) - q * np.log2(q)
    return gain(df, name, boundary, target_col) / tv


# Gini
def gini(series):
    p = series.mean()
    q = 1 - p
    return 1 - p ** 2 - q ** 2


# Gini指数, 实际返回1 - gini_index
def neg_gini_index(df, name, boundary, target_col='target'):
    p1 = df.loc[df[name] > boundary, target_col].size / df[target_col].size
    p2 = 1 - p1
    gini_ = p1 * gini(df[df[name] > boundary][target_col]) + p2 * gini(df[df[name] <= boundary][target_col])
    return 1 - gini_


class CustomDecisionTree(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_impurity=0.0, alpha=0.0, random_state=None, criterion='gini',
                 verbose=0, progress_interval=10):
        # 初始化参数
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity = min_impurity
        self.alpha = alpha
        self.random_state = random_state
        self.criterion = criterion
        self.verbose = verbose
        self.progress_interval = progress_interval
        self.tree_ = None
        self.feature_names_ = None
        self.target_col_ = 'target'
        self.node_count = 0
        self.start_time = None
        self.total_nodes_estimate = 0

        # 设置日志
        self.logger = logging.getLogger("CustomDecisionTree")
        if verbose > 0:
            logging.basicConfig(level=logging.INFO)

        # 设置分裂函数
        if criterion == 'gini':
            self.split_func = neg_gini_index
        elif criterion == 'entropy':
            self.split_func = gain
        elif criterion == 'gain_ratio':
            self.split_func = gain_rate
        else:
            raise ValueError("criterion must be 'gini', 'entropy', or 'gain_ratio'")


    def fit(self, X, y):
        """训练决策树"""
        # 记录开始时间
        fit_start_time = time.time()

        if self.verbose > 0:
            print(f"开始训练决策树 - 参数: max_depth={self.max_depth}, criterion={self.criterion}")

        self.start_time = time.time()
        self.node_count = 0

        # 估计总节点数
        if self.max_depth is not None:
            self.total_nodes_estimate = 2 ** (self.max_depth + 1) - 1
        else:
            n_features = X.shape[1] if hasattr(X, 'shape') else len(X.columns)
            self.total_nodes_estimate = n_features * 10
        # 设置随机种子
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # 转换输入为DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]
            X.columns = self.feature_names_
        else:
            self.feature_names_ = X.columns.tolist()

        # 创建数据副本并添加目标列和权重
        data = X.copy()
        data[self.target_col_] = y
        data['weight'] = 1.0

        # 调用建树函数
        hargs = (self.max_depth, self.min_samples_split, self.min_impurity, self.alpha, self.min_samples_leaf)
        self.tree_ = self._build_tree(data, self.split_func, hargs=hargs)

        if self.verbose > 0:
            elapsed_time = time.time() - self.start_time
            print(f"训练完成，耗时: {elapsed_time:.2f} 秒，共构建 {self.node_count} 个节点")

            # 计算拟合时间
        fit_time = time.time() - fit_start_time

        # 更新全局进度
        progress_tracker = ProgressTracker()
        progress_tracker.update_progress(self.get_params(), fit_time)
        return self

    def predict(self, X):
        """预测方法 - 返回离散的类别标签"""
        # 获取概率预测
        probas = self.predict_proba(X)
        # 将概率转换为类别标签（阈值0.5）
        return (probas[:, 1] > 0.5).astype(int)

    def predict_proba(self, X):
        """预测概率方法"""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            if self.feature_names_ is not None:
                X.columns = self.feature_names_

        probas = []
        for _, sample in X.iterrows():
            proba = self._predict_sample(sample, self.tree_)
            probas.append([1 - proba, proba])
        return np.array(probas)

    def score(self, X, y, **kwargs):
        """评分方法（默认使用准确率）
        :param **kwargs:
        """
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def get_params(self, deep=True):
        """获取参数（GridSearchCV需要）"""
        return {
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'min_impurity': self.min_impurity,
            'alpha': self.alpha,
            'random_state': self.random_state,
            'criterion': self.criterion
        }

    def set_params(self, **params):
        """设置参数（GridSearchCV需要）"""
        for param, value in params.items():
            setattr(self, param, value)
        return self

    def _choose_boundary(self, data, name, f):
        """选择最佳分类点"""
        best_boundary = 0
        best_gain = 0
        # 连续值
        q = 10
        if data[name].nunique(dropna=True) > q:
            bins = pd.qcut(data[name], q, retbins=True, duplicates='drop')[1]
            for i in range(1, np.size(bins) - 2):
                if f(data, name, bins[i], self.target_col_) > best_gain:
                    best_boundary = bins[i]
                    best_gain = f(data, name, best_boundary, self.target_col_)
        # 0 or 1
        elif data[name].nunique(dropna=True) == 2:
            best_boundary = 0.5
            best_gain = f(data, name, 0.5, self.target_col_)
        # 离散值
        else:
            val_list = np.unique(data[name])
            boundary_list = (val_list[1:] + val_list[:-1]) / 2
            best_gain = 0
            for b in boundary_list:
                if f(data, name, b, self.target_col_) > best_gain:
                    best_gain = f(data, name, b, self.target_col_)
                    best_boundary = b
        return best_boundary, best_gain

    def _choose_feature(self, data, f):
        """选择最佳分类特征"""
        best_gain = 0
        best_feature = None
        best_boundary = 0
        for name in data.columns:
            if name not in [self.target_col_, 'weight']:
                boundary, gain_val = self._choose_boundary(data, name, f)
                if gain_val > best_gain:
                    best_gain = gain_val
                    best_feature = name
                    best_boundary = boundary
        return best_feature, best_boundary, best_gain

    def _possess(self, data, nan_data, name, f, hargs, depth):
        """分裂后处理"""
        data_ = pd.concat([data, nan_data])
        if data[name].nunique(dropna=True) == 1:
            data_.drop([name], axis=1)
        if (data_[self.target_col_].nunique(dropna=True) == 1 or
                len(data_.columns) == 2 or  # 只剩下目标列和权重列
                len(data) < 5):
            return self._cal_label(data_)
        else:
            return self._build_tree(data_, f, hargs=hargs, depth=depth + 1)

    def _build_tree(self, data, f, hargs, depth=0):
        """建树算法实现"""
        # 增加节点计数
        self.node_count += 1

        # 定期报告进度（仅在详细模式下）
        if self.verbose > 0 and self.node_count % self.progress_interval == 0:
            elapsed = time.time() - self.start_time
            progress = min(1.0, self.node_count / self.total_nodes_estimate)
            estimated_total = elapsed / progress if progress > 0 else float('inf')
            remaining = estimated_total - elapsed

            print(f"节点: {self.node_count}/{self.total_nodes_estimate} "
                  f"({progress * 100:.1f}%) - 时间: {elapsed:.1f}s "
                  f"[剩余: {remaining:.1f}s]")

        name, boundary, gain_val = self._choose_feature(data, f)

        # 提前终止条件
        n_samples = len(data)
        total_weight = data['weight'].sum()

        # 更多终止条件
        max_depth, min_samples_split, min_impurity, alpha, min_samples_leaf = hargs
        stop_conditions = [
            max_depth is not None and depth >= max_depth,  # 最大深度
            n_samples < min_samples_leaf,  # 最小样本数
            total_weight < min_samples_split,  # 最小权重和
            data[self.target_col_].nunique() == 1,  # 纯节点
            self._cal_impurity(data) < min_impurity  # 杂质足够低
        ]

        if any(stop_conditions):
            return self._cal_label(data)
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

            false_dict = self._possess(data1, data3_f, name, f, hargs, depth)
            true_dict = self._possess(data2, data3_t, name, f, hargs, depth)
            branch = {
                'feature': name,
                'boundary': boundary,
                'False': false_dict,
                'True': true_dict,
                'weight': weight
            }
        return branch

    def _cal_label(self, data):
        """计算标签"""
        w_sum = data['weight'].sum() + 1e-10
        prob = np.sum(np.array(data['weight']) * np.array(data[self.target_col_])) / w_sum
        return prob

    def _cal_impurity(self, data):
        """计算杂质"""
        prob = self._cal_label(data)
        return min(prob, 1 - prob)

    def _predict_sample(self, sample, tree):
        """使用树结构预测单个样本"""
        if not self._is_leaf(tree):
            feature = tree['feature']
            boundary = tree['boundary']
            weight = tree['weight']  # weight = true rate

            if pd.isna(sample[feature]):
                res = (weight * self._predict_sample(sample, tree['True']) +
                       (1 - weight) * self._predict_sample(sample, tree['False']))
            else:
                if sample[feature] > boundary:
                    res = self._predict_sample(sample, tree['True'])
                else:
                    res = self._predict_sample(sample, tree['False'])
        else:
            res = tree
        return res

    def _is_leaf(self, tree):
        """检查是否为叶节点"""
        return not isinstance(tree, dict)

    def _cut_branch(self, data, tree, alpha):
        """剪枝"""
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

        if not self._is_leaf(tree['True']):
            tree['True'] = self._cut_branch(data_t, tree['True'], alpha)
        if not self._is_leaf(tree['False']):
            tree['False'] = self._cut_branch(data_f, tree['False'], alpha)

        if self._is_leaf(tree['True']) and self._is_leaf(tree['False']):
            impurity0 = self._cal_impurity(data)
            impurity_t = self._cal_impurity(data_t)
            impurity_f = self._cal_impurity(data_f)
            if impurity0 + alpha <= (1 - weight) * impurity_f + weight * impurity_t:
                tree = self._cal_label(data)
        return tree

    def prune(self, X, y):
        """剪枝方法"""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            if self.feature_names_ is not None:
                X.columns = self.feature_names_

        data = X.copy()
        data[self.target_col_] = y
        data['weight'] = 1.0

        self.tree_ = self._cut_branch(data, self.tree_, self.alpha)
        return self


# 使用示例
if __name__ == '__main__':
    from sklearn.datasets import make_classification
    from sklearn.model_selection import GridSearchCV, train_test_split

    # 创建示例数据
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 定义参数网格
    param_grid = {
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 5, 10],
        'min_impurity': [0.0, 0.1, 0.2],
        'alpha': [0.0, 0.1, 0.2],
        'criterion': ['gini', 'entropy']
    }

    # 计算总拟合次数
    n_splits = 5  # 交叉验证折数
    n_params = len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * \
               len(param_grid['min_samples_leaf']) * len(param_grid['min_impurity']) * \
            len(param_grid['alpha']) * len(param_grid['criterion'])
    total_fits = n_splits * n_params

    # 初始化进度跟踪器
    n_jobs = 4
    progress_tracker = ProgressTracker()
    progress_tracker.init_progress(total_fits, n_jobs=n_jobs)

    # 创建 GridSearchCV
    grid_search = GridSearchCV(
        estimator=CustomDecisionTree(verbose=0),  # 关闭单个决策树的详细输出
        param_grid=param_grid,
        scoring='accuracy',
        cv=5,
        n_jobs=n_jobs,
        verbose=0  # 关闭 GridSearchCV 的默认输出
    )

    # 执行网格搜索
    grid_search.fit(X_train, y_train)

    # 完成进度跟踪
    progress_tracker.finish()

    # 输出最佳参数和得分
    print("最佳参数:", grid_search.best_params_)
    print("最佳交叉验证得分:", grid_search.best_score_)
    print("测试集得分:", grid_search.score(X_test, y_test))

    # 使用最佳模型进行预测
    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X_test)
    print("预测结果示例:", predictions[:10])
    '''
    Fitting 5 folds for each of 1440 candidates, totalling 7200 fits
    最佳参数: {'alpha': 0.0, 'criterion': 'gini', 'max_depth': 5, 'min_impurity': 0.1, 'min_samples_leaf': 1, 'min_samples_split': 20}
    最佳交叉验证得分: 0.8975
    测试集得分: 0.905
    预测结果示例: [1 1 0 1 1 0 0 1 0 0]

    '''