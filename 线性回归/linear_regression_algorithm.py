# 线性回归
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
def test():
    # 设置随机种子
    np.random.seed(int(time.perf_counter_ns()%1000))

    # 线性模型参量

    # 维度
    N=2

    # 数据集大小
    size=1000

    # 噪声强度
    noise_power=10

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

    # 计算

    # 初始化权重
    theta = np.random.rand(X_train.shape[1] + 1)*1  # 包括偏置项，所以是11个权重
    theta_hajime=theta

    # 设置学习率
    alpha = 5*1e-6

    # 设置迭代次数
    n_iterations = 10000

    # 添加偏置项到 X_train
    X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]

    # 梯度下降算法
    loss_last=9999999999
    counter=0
    for iteration in range(n_iterations):
        gradients = X_train_b.T.dot(X_train_b.dot(theta) - y_train)/size
        theta = theta - alpha * gradients
        
        loss=np.mean((X_test_b.dot(theta) - y_test)**2)
        if(abs(loss-loss_last)/loss<0.0005):
            break
        loss_last=loss
        #print(f"第{iteration}次迭代：theta{theta}\n,gradients:{gradients}\n,loss:{loss}\n")
        #print(f"第{iteration}次迭代：loss:{loss}")

    # 打印最终的权重
    print(f"模型权重：{weight}")
    print("最终的权重:", theta)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(X[:,0],X[:,1],y)
    

    # X1 = np.linspace(0, 500, 100)
    # Y1 = np.linspace(0, 500, 100)
    # X1, Y1 = np.meshgrid(X1, Y1)
    # Z1 = X1*theta[1]+Y1*theta[2]+theta[0]
    # print(Z1)
    # #ax1=fig.add_subplot(212, projection='3d')
    # ax.plot_surface(X1, Y1, Z1, cmap='viridis')
    # plt.show()

    if ((theta[1]-weight[0])/weight[0]<0.001) and ((theta[2]-weight[1])/weight[0]<0.01):
        return True
    else:
        return False
    
cnt=0
for i in range(100):
    if(test()):
        cnt+=1

print(f"准确率{cnt/100*100}%")