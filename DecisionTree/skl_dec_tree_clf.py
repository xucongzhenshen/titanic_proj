import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class GridSearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GridSearchCV参数调优")
        self.root.geometry("900x700")

        # 创建示例数据
        self.X, self.y = make_classification(
            n_samples=1000, n_features=20, n_informative=15,
            n_redundant=5, random_state=42
        )

        # 划分训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        self.create_widgets()

    def create_widgets(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 参数设置框架
        param_frame = ttk.LabelFrame(main_frame, text="参数网格设置", padding="5")
        param_frame.pack(fill=tk.X, pady=5)

        # 最大深度
        ttk.Label(param_frame, text="最大深度:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.max_depth_entry = ttk.Entry(param_frame, width=30)
        self.max_depth_entry.insert(0, "3,5,7,10,15,20,None")
        self.max_depth_entry.grid(row=0, column=1, padx=5, pady=2)

        # 最小样本分割
        ttk.Label(param_frame, text="最小样本分割:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.min_samples_split_entry = ttk.Entry(param_frame, width=30)
        self.min_samples_split_entry.insert(0, "2,5,10,15")
        self.min_samples_split_entry.grid(row=1, column=1, padx=5, pady=2)

        # 最小样本叶节点
        ttk.Label(param_frame, text="最小样本叶节点:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.min_samples_leaf_entry = ttk.Entry(param_frame, width=30)
        self.min_samples_leaf_entry.insert(0, "1,2,5,10")
        self.min_samples_leaf_entry.grid(row=2, column=1, padx=5, pady=2)

        # 分裂标准
        ttk.Label(param_frame, text="分裂标准:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.criterion_entry = ttk.Entry(param_frame, width=30)
        self.criterion_entry.insert(0, "gini,entropy")
        self.criterion_entry.grid(row=3, column=1, padx=5, pady=2)

        # 交叉验证折数
        ttk.Label(param_frame, text="交叉验证折数:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        self.cv_entry = ttk.Entry(param_frame, width=30)
        self.cv_entry.insert(0, "5")
        self.cv_entry.grid(row=4, column=1, padx=5, pady=2)

        # 并行工作数
        ttk.Label(param_frame, text="并行工作数(-1为全部):").grid(row=5, column=0, sticky=tk.W, padx=5, pady=2)
        self.n_jobs_entry = ttk.Entry(param_frame, width=30)
        self.n_jobs_entry.insert(0, "-1")
        self.n_jobs_entry.grid(row=5, column=1, padx=5, pady=2)

        # 按钮框架
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)

        ttk.Button(button_frame, text="运行GridSearchCV", command=self.run_grid_search).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="清除结果", command=self.clear_results).pack(side=tk.LEFT, padx=5)

        # 结果框架
        results_frame = ttk.LabelFrame(main_frame, text="结果", padding="5")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # 文本区域显示结果
        self.results_text = tk.Text(results_frame, height=15)
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)

        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def clear_results(self):
        self.results_text.delete(1.0, tk.END)

    def run_grid_search(self):
        try:
            # 解析参数
            max_depth = [None if x == 'None' else int(x) for x in self.max_depth_entry.get().split(',')]
            min_samples_split = [int(x) for x in self.min_samples_split_entry.get().split(',')]
            min_samples_leaf = [int(x) for x in self.min_samples_leaf_entry.get().split(',')]
            criterion = self.criterion_entry.get().split(',')
            cv = int(self.cv_entry.get())
            n_jobs = int(self.n_jobs_entry.get())

            # 创建参数网格
            param_grid = {
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'criterion': criterion
            }

            self.status_var.set("正在运行GridSearchCV...")
            self.root.update()

            # 创建分类器
            dt_classifier = DecisionTreeClassifier(random_state=42)

            # 记录开始时间
            start_time = time.time()

            # 创建GridSearchCV对象
            grid_search = GridSearchCV(
                estimator=dt_classifier,
                param_grid=param_grid,
                scoring='accuracy',
                cv=cv,
                n_jobs=n_jobs,
                verbose=1
            )

            # 执行网格搜索
            grid_search.fit(self.X_train, self.y_train)

            # 计算耗时
            elapsed_time = time.time() - start_time

            # 显示结果
            self.results_text.insert(tk.END, f"GridSearchCV完成，耗时: {elapsed_time:.2f}秒\n")
            self.results_text.insert(tk.END, f"尝试的参数组合数: {len(grid_search.cv_results_['params'])}\n")
            self.results_text.insert(tk.END, f"最佳参数: {grid_search.best_params_}\n")
            self.results_text.insert(tk.END, f"最佳交叉验证得分: {grid_search.best_score_:.4f}\n")

            # 使用最佳参数评估测试集
            best_model = grid_search.best_estimator_
            test_score = best_model.score(self.X_test, self.y_test)
            self.results_text.insert(tk.END, f"测试集得分: {test_score:.4f}\n\n")

            # 显示前10个最佳参数组合
            self.results_text.insert(tk.END, "排名前10的参数组合:\n")
            results_df = pd.DataFrame(grid_search.cv_results_)
            top_10 = results_df.nlargest(10, 'mean_test_score')

            for i, (_, row) in enumerate(top_10.iterrows(), 1):
                self.results_text.insert(tk.END, f"{i}. 参数: {row['params']}, 得分: {row['mean_test_score']:.4f}\n")

            self.results_text.see(tk.END)
            self.status_var.set("完成")

            # 可视化结果
            self.visualize_results(grid_search)

        except Exception as e:
            self.results_text.insert(tk.END, f"错误: {str(e)}\n")
            self.status_var.set("错误发生")

    def visualize_results(self, grid_search):
        # 创建一个新窗口来显示可视化结果
        viz_window = tk.Toplevel(self.root)
        viz_window.title("GridSearchCV可视化结果")
        viz_window.geometry("800x600")

        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('GridSearchCV参数调优结果', fontsize=16)

        # 提取结果
        results = pd.DataFrame(grid_search.cv_results_)

        # 1. 参数重要性热图
        param_columns = [col for col in results.columns if col.startswith('param_')]
        param_importance = results[param_columns + ['mean_test_score']].copy()

        # 简化参数值用于可视化
        for col in param_columns:
            param_importance[col] = param_importance[col].astype(str)

        # 创建交叉表显示参数组合与得分的关系
        if len(param_columns) >= 2:
            pivot_table = pd.pivot_table(
                param_importance,
                values='mean_test_score',
                index=param_columns[0],
                columns=param_columns[1],
                aggfunc='mean'
            )
            sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[0, 0])
            axes[0, 0].set_title('参数组合热图')

        # 2. 不同参数值的得分分布
        if 'param_max_depth' in results.columns:
            sns.boxplot(x='param_max_depth', y='mean_test_score', data=results, ax=axes[0, 1])
            axes[0, 1].set_title('不同最大深度的得分分布')
            axes[0, 1].tick_params(axis='x', rotation=45)

        # 3. 训练时间与得分的关系
        if 'mean_fit_time' in results.columns:
            axes[1, 0].scatter(results['mean_fit_time'], results['mean_test_score'])
            axes[1, 0].set_xlabel('平均训练时间(秒)')
            axes[1, 0].set_ylabel('平均测试得分')
            axes[1, 0].set_title('训练时间与得分的关系')

        # 4. 参数组合的得分分布
        top_10 = results.nlargest(10, 'mean_test_score')
        axes[1, 1].barh(range(len(top_10)), top_10['mean_test_score'])
        axes[1, 1].set_yticks(range(len(top_10)))

        # 简化参数标签
        param_labels = []
        for _, row in top_10.iterrows():
            label_parts = []
            for param in param_columns:
                param_name = param.replace('param_', '')
                param_value = str(row[param])
                label_parts.append(f"{param_name}={param_value}")
            param_labels.append(", ".join(label_parts))

        axes[1, 1].set_yticklabels(param_labels, fontsize=8)
        axes[1, 1].set_xlabel('平均测试得分')
        axes[1, 1].set_title('前10个最佳参数组合')

        plt.tight_layout()

        # 将图表嵌入Tkinter窗口
        canvas = FigureCanvasTkAgg(fig, master=viz_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 添加关闭按钮
        ttk.Button(viz_window, text="关闭", command=viz_window.destroy).pack(pady=10)


if __name__ == "__main__":
    root = tk.Tk()
    app = GridSearchApp(root)
    root.mainloop()