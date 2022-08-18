import numpy as np

# 固定p优化q

# p: n×1
def ln(q) : # 矩阵内每个元素分别求ln
    x = np.zeros([n, 1])
    i = 0
    while i < n :
        if q[i] <= 0 :
            x[i] = -10000.0
        else :
            x[i] = np.log(q[i])
        i += 1
    return x

def dev(p, q) : # pq每个元素分别相除
    x = np.zeros([n, 1])
    for i in range(n) :
        x[i] = p[i] / q[i]
        i += 1
    return x

def ori_obj(p, q) : # 原目标函数
    return np.matmul(p.T, ln(q))

# A =                    b = 
# [ 1,  1, ... ,  1      [1
#  -1,  0, ... ,  0       0
#   0, -1, ... ,  0       0
#   ...                   .
#   0,  0, ... , -1]      0]
def LR(p, q, u) : # 拉格朗日松弛函数
    temp_num = np.matmul(A, q) - b # Aq - b
    return ori_obj(p, q) + np.matmul(u.T, temp_num)

def LR_grad_q(p, q, u) : # 拉格朗日松弛函数关于q求梯度
    return dev(p, q) + np.matmul(u.T, A).T

def LR_grad_u(q) : # 拉格朗日松弛函数关于u求梯度
    return np.matmul(A, q) - b

def judge_u(u) : # 限定拉格朗日乘子u各元素的正负
    i = 1
    while i < n+1 :
        if u[i] > 0 :
            u[i] = 0
        i += 1

n = 20
A1 = np.ones([1, n])
A2 = -np.identity(n)
A = np.append(A1, A2, axis=0)
del A1, A2
b = np.append(np.ones([1,1]), np.zeros([n,1]), axis=0)

p = np.ones([n, 1]) - np.random.rand(n, 1)
p /= p.sum()
q = np.ones([n, 1]) - np.random.rand(n, 1)
print("init_q = ", q.T)
u = np.zeros([n+1, 1])
a1 = 0.01
a2 = 0.01
d = 0.00005 # 梯度下降终止条件 |LR(p, q_new, u) - LR(p, q, u)| < d
while True :
    while True :
        q_new = q + a1*LR_grad_q(p, q, u)
        if LR(p, q_new, u) < LR(p, q, u) : # 更新a1
            a1 /= 2
        if np.sum(np.abs(LR(p, q_new, u) - LR(p, q, u))) > d : # 未到终止条件继续更新
            q = q_new
            continue
        q = q_new
        break
    u_new = u - a2*LR_grad_u(q)
    judge_u(u_new)
    if LR(p, q, u_new) > LR(p, q, u) : # 更新a2
        a2 /= 2
    if np.sum(np.abs(LR(p, q, u_new) - LR(p, q, u))) > d :
        u = u_new
        continue
    u = u_new
    break
print("q = ", q.T)
print("p = ", p.T)
# Error
Error = 0
error = np.abs(p-q)
for i in range(n) :
    Error += error[i]
Error *= n
print('Error = %.6f'%(Error))