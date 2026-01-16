import numpy as np
#####################################
#引入第6章核心：把更新规则抽象成 Optimizer（SGD / Momentum / Adam）
#######################################
# -------------------------
# Utils
# -------------------------带类型注解的定义
def softmax(x: np.ndarray) -> np.ndarray:
    #softmax(参数x: 表示参数的类型) -> :表示函数的返回值类型
    # x: (batch, num_classes) or (num_classes,)
    if x.ndim == 2:
        x = x - np.max(x, axis=1, keepdims=True)  #防止溢出
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

def cross_entropy_error(y: np.ndarray, t: np.ndarray) -> float:
    """
    y: (batch, C) probabilities
    t: (batch,) label indices or (batch, C) one-hot
    """
    if y.ndim == 1:
        y = y.reshape(1, -1)
        t = t.reshape(1, -1)

    batch_size = y.shape[0]

    # if t is one-hot -> convert to label indices
    if t.ndim == 2 and t.shape == y.shape:
        t_idx = np.argmax(t, axis=1)
    else:
        t_idx = t.astype(int)
    
    t_idx = np.array(t_idx).reshape(batch_size,)

    # avoid log(0)
    eps = 1e-7
    return -np.sum(np.log(y[np.arange(batch_size), t_idx] + eps)) / batch_size

def numerical_gradient(f, x: np.ndarray) -> np.ndarray:
    h = 1e-5
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        tmp = x[idx]

        x[idx] = tmp + h
        fxh1 = f()

        x[idx] = tmp - h
        fxh2 = f()

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp
        it.iternext()

    return grad

# -------------------------
# Layers
# -------------------------
class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dout = dout.copy()
        dout[self.mask] = 0
        dx = dout
        return dx

class Affine:
    def __init__(self, W: np.ndarray, b: np.ndarray):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        out = x @ self.W + self.b
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dx = dout @ self.W.T
        self.dW = self.x.T @ dout
        self.db = np.sum(dout, axis=0)
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None  # softmax output
        self.t = None  # labels (index or one-hot)

    def forward(self, x: np.ndarray, t: np.ndarray) -> float:
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout: float = 1.0) -> np.ndarray:
        batch_size = self.y.shape[0]

        # convert one-hot to index if needed
        if self.t.ndim == 2 and self.t.shape == self.y.shape:
            t_idx = np.argmax(self.t, axis=1)
        else:
            t_idx = self.t.astype(int)
        
        t_idx = np.array(t_idx).reshape(batch_size,)

        dx = self.y.copy()
        dx[np.arange(batch_size), t_idx] -= 1
        dx = (dx / batch_size) * dout
        return dx

class BatchNorm:
    def __init__(self, gamma, beta, momentum=0.9, eps=1e-5):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.eps = eps

        # running stats for inference
        self.running_mean = None
        self.running_var = None

        # cache for backward
        self.batch_size = None
        self.xc = None
        self.std = None
        self.xn = None

        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        if self.running_mean is None:
            D = x.shape[1]
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flg:
            mu = np.mean(x, axis=0)
            xc = x - mu
            var = np.mean(xc ** 2, axis=0)
            std = np.sqrt(var + self.eps)
            xn = xc / std

            # cache
            self.batch_size = x.shape[0]
            self.xc = xc
            self.std = std
            self.xn = xn

            # update running stats
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            xc = x - self.running_mean
            xn = xc / np.sqrt(self.running_var + self.eps)

        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout):
        # dout: (N, D)
        self.dbeta = np.sum(dout, axis=0)
        self.dgamma = np.sum(dout * self.xn, axis=0)

        dxn = dout * self.gamma  # (N, D)

        # x̂ = xc / std
        dxc = dxn / self.std  # (N, D)
        dstd = -np.sum((dxn * self.xc) / (self.std ** 2), axis=0)  # (D,)

        # std = sqrt(var + eps)
        dvar = 0.5 * dstd / self.std  # (D,)

        # var = mean(xc^2)
        dxc += (2.0 / self.batch_size) * self.xc * dvar  # (N, D)

        # xc = x - mu, and mu = mean(x)
        dmu = np.sum(dxc, axis=0)  # (D,)
        dx = dxc - dmu / self.batch_size  # (N, D)

        return dx


# -------------------------
# Network
# -------------------------
class TwoLayerNet:
     #网络支持三种初始化
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 init: str = "std", weight_init_std: float = 0.01, seed: int = 0):
        rng = np.random.default_rng(seed)#随机数生成器,独立的随机数生成器,不影响全局随机状态.
        self.params = {}
        """
        两层神经网络的初始化函数
        参数：
            input_size: 输入层神经元数量（特征维度）
            hidden_size: 隐藏层神经元数量,控制模型容量
            output_size: 输出层神经元数量（分类数量）
            init: 权重初始化方法，影响训练速度和稳定性,可选："std", "xavier", "he"
            weight_init_std: 标准初始化方法的缩放系数(默认0.01)
            seed: 随机数种子，确保结果可重复
        """

        # 选择初始化尺度,初始化策略选择
        if init == "std":
            W1_scale = weight_init_std
            W2_scale = weight_init_std
        elif init == "xavier":
            W1_scale = np.sqrt(1.0 / input_size)
            W2_scale = np.sqrt(1.0 / hidden_size)
        elif init == "he":
            W1_scale = np.sqrt(2.0 / input_size)
            W2_scale = np.sqrt(2.0 / hidden_size)
        else:
            raise ValueError("init must be one of: 'std', 'xavier', 'he'")

        # 参数
        self.params["W1"] = W1_scale * rng.standard_normal((input_size, hidden_size))
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = W2_scale * rng.standard_normal((hidden_size, output_size))
        self.params["b2"] = np.zeros(output_size)
        self.params["gamma1"] = np.ones(hidden_size)
        self.params["beta1"] = np.zeros(hidden_size)

        # layers (forward order)
        self.layers = {}
        self.layers["Affine1"] = Affine(self.params["W1"], self.params["b1"])
        self.layers["BatchNorm1"] = BatchNorm(self.params["gamma1"], self.params["beta1"])
        self.layers["ReLU1"] = ReLU()
        self.layers["Affine2"] = Affine(self.params["W2"], self.params["b2"])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x: np.ndarray, train_flg=True) -> np.ndarray: #神经网络模型的预测方法
        for layer in self.layers.values(): #按顺序遍历网络所有层（self.layers字典的值）
            # 如果层支持 train_flg（比如 BatchNorm/Dropout），就传进去
            if hasattr(layer, "forward") and layer.forward.__code__.co_argcount >= 3:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x: np.ndarray, t: np.ndarray, train_flg=True) -> float:
        y = self.predict(x, train_flg=train_flg) #执行前向传播，获取模型对输入x的预测结果y
        return self.lastLayer.forward(y, t) #调用预设的损失层（如SoftmaxWithLoss），计算预测值y与真实标签t之间的损失值并返回

#计算神经网络预测准确率。
    def accuracy(self, x: np.ndarray, t: np.ndarray) -> float:
        y = self.predict(x, train_flg=False) #获取预测结果
        y_label = np.argmax(y, axis=1)#取概率最大索引作为预测标签
        if t.ndim == 2: # 若真实标签t为one-hot编码（二维），则同样转为索引形式，否则直接使用。
            t_label = np.argmax(t, axis=1)
        else:
            t_label = t
        return float(np.mean(y_label == t_label))

    def gradient(self, x: np.ndarray, t: np.ndarray) -> dict: #实现神经网络的反向传播计算梯度
        # forward
        self.loss(x, t, train_flg=True)

        # backward
        dout = 1.0
        dout = self.lastLayer.backward(dout)

        # reverse order，从输出层开始反向传播，逐层计算梯度
        for layer in reversed(list(self.layers.values())):
            dout = layer.backward(dout)

        grads = { #按网络层顺序（Affine1/Affine2）收集权重和偏置的梯度值
            "W1": self.layers["Affine1"].dW,
            "b1": self.layers["Affine1"].db,
            "W2": self.layers["Affine2"].dW,
            "b2": self.layers["Affine2"].db,
            "gamma1": self.layers["BatchNorm1"].dgamma,
            "beta1": self.layers["BatchNorm1"].dbeta,
        }
        return grads
    
    def numerical_gradients(self, x: np.ndarray, t: np.ndarray) -> dict: #计算两层神经网络各参数（W1/b1/W2/b2）的数值梯度。
        grads = {}   
        # 检查是否有BatchNorm层并保存其运行均值和方差
        bn_layers = []
        for name, layer in self.layers.items():
            if "BatchNorm" in name:
                #保存当前运行均值和方差副本
                if layer.running_mean is  None: #如果运行均值和方差为None
                    D = self.params["gamma1"].shape[0]
                    layer.running_mean = np.zeros(D)
                    layer.running_var = np.zeros(D)
                bn_layers.append((layer, layer.running_mean.copy(), layer.running_var.copy()))


        #逐个参数计算数值梯度 注意：数值梯度会直接改 params，所以要逐个算
        for key in ["W1", "b1", "W2", "b2", "gamma1", "beta1" ]:
            def f():
                # 在计算损失前恢复 BatchNorm 的运行均值和方差
                for bn_layer, saved_mean, saved_var in bn_layers:
                    bn_layer.running_mean = saved_mean.copy()
                    bn_layer.running_var = saved_var.copy()
                return self.loss(x, t, train_flg=True)
            grads[key] = numerical_gradient(f, self.params[key])
        #恢复 BatchNorm 层的原始运行均值和方差（保持网络状态不变）
        for bn_layer, saved_mean, saved_var in bn_layers:
            bn_layer.running_mean = saved_mean
            bn_layer.running_var = saved_var

        return grads



# -------------------------
# Gradient Check (unit test)
# ------------------------- 
# 用于梯度检验，验证神经网络反向传播计算的梯度是否正确。
def gradient_check():
    np.random.seed(0) #固定随机种子确保可以复现
    # 初始化两层神经网络（4输入-5隐藏-3输出）
    net = TwoLayerNet(input_size=4, hidden_size=5, output_size=3, weight_init_std=0.01, seed=0)

    x = np.random.randn(8, 4)               # batch=2，生成2个4维随机输入样本
    t = np.random.randint(0,3 ,size=8)                    # label index (0..C-1)，对应样本的真实标签（0和1类）

    grad_backprop = net.gradient(x, t)      # 反向传播计算解析梯度
    grad_numerical = net.numerical_gradients(x, t) # 数值法计算近似梯度

     # 逐参数对比两种梯度计算结果
    for k in grad_backprop.keys():
        diff = np.mean(np.abs(grad_backprop[k] - grad_numerical[k]))  # 计算平均绝对误差
        print(f"{k}: mean abs diff = {diff:.8f}")                     # 输出误差值（理想应<1e-8）

#-------------------------------------------
#先加三个优化器类（最小实现）
#-------------------------------------------
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for k in params:
            params[k] -= self.lr * grads[k]

class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {k: np.zeros_like(params[k]) for k in params}
        for k in params:
            self.v[k] = self.momentum * self.v[k] - self.lr * grads[k]
            params[k] += self.v[k]

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m = {k: np.zeros_like(params[k]) for k in params}
            self.v = {k: np.zeros_like(params[k]) for k in params}

        self.iter += 1
        lr_t = self.lr * (np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter))

        for k in params:
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grads[k]
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (grads[k] ** 2)
            params[k] -= lr_t * self.m[k] / (np.sqrt(self.v[k]) + self.eps)
           
#--------------------------
#随机数据
#--------------------------
def run_compare_optimizers():
    np.random.seed(0)

    # data
    N = 200
    x = np.random.randn(N, 4)
    true_W = np.array([[1.0, -1.0, 0.5],
                       [0.3,  0.2, -0.7],
                       [-0.6, 0.4,  0.1],
                       [0.2,  0.8, -0.2]])
    logits = x @ true_W
    t = np.argmax(logits, axis=1)

    configs = [
        ("SGD(lr=0.5)",      lambda: SGD(lr=0.5)),#把0.1改为0.5，第五次改动.
        ("Momentum(0.03,0.9)",lambda: Momentum(lr=0.03, momentum=0.9)),
        ("Adam(lr=0.003)",    lambda: Adam(lr=0.003)),
    ]

    batch_size = 20
    iters = 200

    for name, opt_fn in configs:
        np.random.seed(0)  # 第四次改动：每次优化器前重置随机种子，确保初始参数和数据顺序相同
        net = TwoLayerNet(input_size=4, hidden_size=10, output_size=3,
                          weight_init_std=0.01, seed=0)
        optimizer = opt_fn()

        best_acc = 0.0
        best_iter = 0

        for it in range(iters):
            batch_idx = np.random.choice(N, batch_size, replace=False)
            xb, tb = x[batch_idx], t[batch_idx]

            grads = net.gradient(xb, tb)

            if it in (0, 100, 101):
                print(">>>", type(optimizer).__name__, "lr =", optimizer.lr)

            # 第三次改动。学习率策略调整，优化器+学习率策略是一套组合拳。
            # 优化器 + 学习率策略：不是换了优化器就完事，而是要配合 schedule。step decay :训练到一半把学习率率降10倍
            if it ==100 and isinstance(optimizer, (SGD, Momentum, Adam)):
                optimizer.lr *= 0.1
            
            optimizer.update(net.params, grads)

            if it % 20 == 0:
                loss = net.loss(x, t)
                acc = net.accuracy(x, t)
                if acc > best_acc:
                    best_acc, best_iter = acc, it
                print(f"{name:18s} | iter {it:03d} | loss {loss:.4f} | acc {acc:.3f}")

        final_loss = net.loss(x, t)
        final_acc  = net.accuracy(x, t)
        print(f"[{name}] best_acc={best_acc:.3f} @iter{best_iter}, final_acc={final_acc:.3f}, final_loss={final_loss:.4f}")
        print("-"*80)




#--------------------------
#新增一个对比初始化的函数
#--------------------------

def compare_inits_with_sgd():
    # 造 toy 数据（与你之前一致）
    np.random.seed(0)
    N = 200
    x = np.random.randn(N, 4)
    true_W = np.array([[1.0, -1.0, 0.5],
                       [0.3,  0.2, -0.7],
                       [-0.6, 0.4,  0.1],
                       [0.2,  0.8, -0.2]])
    logits = x @ true_W
    t = np.argmax(logits, axis=1)

    batch_size = 20
    iters = 200

    for init in ["std", "xavier", "he"]:
        # 关键：保证每种 init 用同一条 batch 抽样序列（公平对比）
        np.random.seed(0)

        net = TwoLayerNet(input_size=4, hidden_size=10, output_size=3,
                          init=init, weight_init_std=0.01, seed=0)
        optimizer = SGD(lr=0.5)  # 你刚刚验证过这个 SGD lr 是好用的

        best_acc, best_it = 0.0, 0

        for it in range(iters):
            batch_idx = np.random.choice(N, batch_size, replace=False)
            xb, tb = x[batch_idx], t[batch_idx]

            grads = net.gradient(xb, tb)
            optimizer.update(net.params, grads)

            if it % 20 == 0:
                loss = net.loss(x, t)
                acc = net.accuracy(x, t)
                if acc > best_acc:
                    best_acc, best_it = acc, it
                print(f"init={init:6s} | iter {it:03d} | loss {loss:.4f} | acc {acc:.3f}")

        final_loss = net.loss(x, t)
        final_acc = net.accuracy(x, t)
        print(f"[init={init}] best_acc={best_acc:.3f}@{best_it}, final_acc={final_acc:.3f}, final_loss={final_loss:.4f}")
        print("-" * 80)



if __name__ == "__main__":
    gradient_check()   
    run_compare_optimizers()      #运行优化器对比实验
    #compare_inits_with_sgd()     # 运行权重初始值初始化对比实验

"""
--------------------------------------------------------
# 第七次改动加了 BN 后,Momentum/Adam 反而变得更抖？怎么调参让它更稳？
1. 修改
    -  加了 BN 后，合适的学习率范围变了，而且动量会放大“过冲”。
    - 把 Momentum 的学习率从 0.1 降到 0.03
    - 把 Adam 也加衰减，看看能否更稳.注意前面设置的打印adam的lr是固定的,所以不会变动。
    - 将ADam 的学习率从 0.003 改成 0.003 
2. 说明
    - BatchNorm 不是“自动变强”的魔法，它改变了合适的学习率/动量区间；配合学习率衰减，训练会更稳定。
    - BN + Momentum 需要更小 lr 或衰减？
        - BN 让每层输入分布在训练中更动态,梯度方向更容易变化,Momentum 的速度项会延续旧方向，容易过冲；因此需要降低 lr 或在后期衰减来稳定收敛。
    - 衰减后步子变小，后期更像“精修”，更可能把 best 往后推。
    - 更平滑 ≠ 更好。你把 Adam 的 lr 降太多后,优化变得“太保守”,结果收敛到一个更差的点(final_acc 0.91,loss 0.465)。


"""
