import numpy as np

# p: n×1
def ln(p) : # 矩阵内每个元素分别求ln
    x = np.zeros([n, 1])
    i = 0
    while i < n :
        x[i] = np.log(p[i])
        i += 1
    return x

def ori_obj(p) : # 原目标函数
    return np.matmul(p.T, ln(p))

# A =                    b = 
# [ 1,  1, ... ,  1      [1
#  -1,  0, ... ,  0       0
#   0, -1, ... ,  0       0
#   ...                   .
#   0,  0, ... , -1]      0]
def LR(p, u) : # 拉格朗日松弛函数
    temp_num = np.matmul(A, p) - b # Ap - b
    return ori_obj(p) + np.matmul(u.T, temp_num)

def LR_grad_p(p, u) : # 拉格朗日松弛函数关于p求梯度
    return ln(p) + np.ones([n, 1]) + np.matmul(u.T, A).T

def LR_grad_u(p, u) : # 拉格朗日松弛函数关于u求梯度
    return np.matmul(A, p) - b

def judge_u(u) : # 限定拉格朗日乘子u各元素的正负
    i = 1
    while i < n+1 :
        if u[i] < 0 :
            u[i] = 0
        i += 1

n = 50
A1 = np.ones([1, n])
A2 = -np.identity(n)
A = np.append(A1, A2, axis=0)
del A1, A2
b = np.append(np.ones([1,1]), np.zeros([n,1]), axis=0)

p = np.ones([n, 1]) - np.random.rand(n, 1)
print("init_p = ", p.T)
u = np.zeros([n+1, 1])
a1 = 0.01
a2 = 0.01
d = 0.0001 # 梯度下降终止条件 |p - p_new| < d
while True :
    while True :
        p_new = p - a1*LR_grad_p(p, u)
        if LR(p_new, u) > LR(p,u): # 更新a1
            a1 /= 2
        if np.sum(np.abs(p - p_new)) > d :
            p = p_new
            continue
        p = p_new
        break
    u_new = u + a2*LR_grad_u(p, u)
    judge_u(u_new)
    if LR(p, u_new) < LR(p, u) : # 更新a2
        a2 /= 2
    if np.sum(np.abs(u - u_new)) > d :
        u = u_new
        continue
    u = u_new
    break
err_abs = abs(np.log(n) + ori_obj(p))
err_rel = -err_abs / ori_obj(p)
print("p = ", p.T)
print("entropy = {:.4f}".format(-ori_obj(p)[0][0]))
print("relative error = {:.4%}".format(err_rel[0][0]))