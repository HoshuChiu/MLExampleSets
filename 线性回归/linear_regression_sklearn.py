import numpy as np
import time
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


np.random.seed(int(time.perf_counter_ns()%1000))

# 线性模型参量

# 维度
N=10

# 数据集大小
size=1000

# 噪声强度
noise_power=100

# 特征强度
x_power=400

# 随机生成权重
weight=np.random.rand(N)*100


# 随机生成特征值
X = np.random.rand(size, N) * x_power

# 合成最终数值
y=np.dot(X,weight)+np.random.randn(size)*noise_power



# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#
linear_model=LinearRegression()

linear_model.fit(X,y)

print(f"模型权重：{weight}")
print("Coefficients:", linear_model.coef_)
print("Intercept:", linear_model.intercept_)