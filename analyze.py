import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

data = pd.read_csv('cleaned_data/train.csv')
x_val = data['Parch'] - data['SibSp']
y_val = data['Age']
x_val = [x + np.random.normal(0, 0.1) for x in x_val]
y_val = [y + np.random.normal(0, 0.1) for y in y_val]
c_val = data['Survived']

plt.style.use('seaborn-v0_8-dark')
fig, ax = plt.subplots()
ax.scatter(x_val, y_val, c=c_val, cmap=plt.cm.Greens, s=2)
# ax.axis([-10,100,0,20])
plt.show()


'''def validate_specific_rule(data):
    # 应用规则筛选数据
    mask = (data['Title2'] > 2.5) & (data['Fare'] > 0) & (data['Is_male'] < 0.5)
    filtered_data = data[mask]
    filtered_target = filtered_data["Survived"]

    if len(filtered_target) > 0:
        # 计算实际死亡率
        actual_mortality = 1 - np.mean(filtered_target)
        print(f"符合规则的数据点数量: {len(filtered_target)}")
        print(f"实际死亡率: {actual_mortality:.4f}")
        print(f"预测死亡率: 0")
        print(f"差异: {abs(1.0 - actual_mortality):.4f}")
    else:
        print("没有数据点符合此规则")


# 使用完整数据集验证
tree_data = pd.read_csv('cleaned_data/tree_train.csv')
validate_specific_rule(tree_data)'''