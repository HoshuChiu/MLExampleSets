import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import time

# 设置模型

## 多项式项数
M=3

## 特征强度
x_power=1000

## 噪声强度
noise_power=5000000

## 数据集大小
size=100000

np.random.seed(int(time.perf_counter_ns()%1000))

# 随机生成权重
coef=np.random.rand(M)

# 随机生成采样点
X=np.random.rand(size)*x_power

# 合成
y=X*coef[0]+X**2*coef[1]+X**3*coef[2]+np.full(size,6e7)+noise_power*np.random.randn(size)

plt.subplot(111)
plt.scatter(X, y, color='blue', s=1)  # 蓝色点，大小为50


X_2=X**2
X_3=X**3

X_ok=np.hstack((X.reshape(-1,1),X_2.reshape(-1,1),X_3.reshape(-1,1)))

print(X_ok.shape)
linear_model=LinearRegression()
linear_model.fit(X_ok,y)

print(f"模型权重：{coef}")
print("Coefficients:", linear_model.coef_)
print("Intercept:", linear_model.intercept_)

x_show=np.linspace(0,1000,1000)
[a,b,c]=linear_model.coef_
y_show=x_show*a+x_show**2*b+x_show**3*c+linear_model.intercept_
plt.scatter(x_show,y_show,color='red',s=1)
plt.show()