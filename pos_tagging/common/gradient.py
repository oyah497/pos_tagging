import numpy as np

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x) # xの属性を引き継いで0で初期化した配列

    # 様々な次元数に対応可能
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val
        it.iternext()
    
    return grad

