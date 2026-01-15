import numpy as np

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

    # avoid log(0)
    eps = 1e-7
    return -np.sum(np.log(y[np.arange(batch_size), t_idx] + eps)) / batch_size

def numerical_gradient(f, x: np.ndarray) -> np.ndarray:
    h = 1e-4
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

        dx = self.y.copy()
        dx[np.arange(batch_size), t_idx] -= 1
        dx = (dx / batch_size) * dout
        return dx

# -------------------------
# Network
# -------------------------
class TwoLayerNet:
        #初始化两层神经网络
        
        #参数:
           # input_size (int): 输入层神经元数量
           # hidden_size (int): 隐藏层神经元数量
           # output_size (int): 输出层神经元数量
           # weight_init_std (float): 权重初始化标准差，默认值为0.01
           # seed (int): 随机数生成器种子，默认值为0
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 weight_init_std: float = 0.01, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.params = {}
        # 初始化网络参数：权重使用正态分布初始化，偏置初始化为零向量
        self.params["W1"] = weight_init_std * rng.standard_normal((input_size, hidden_size))
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * rng.standard_normal((hidden_size, output_size))
        self.params["b2"] = np.zeros(output_size)

        # layers (forward order),构建前向传播层结构：包含两个Affine层和ReLU激活函数
        self.layers = {}
        self.layers["Affine1"] = Affine(self.params["W1"], self.params["b1"])
        self.layers["ReLU1"] = ReLU()
        self.layers["Affine2"] = Affine(self.params["W2"], self.params["b2"])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x: np.ndarray) -> np.ndarray: #神经网络模型的预测方法
        for layer in self.layers.values(): #按顺序遍历网络所有层（self.layers字典的值）
            x = layer.forward(x) # 逐层执行前向传播计算：每层将上一层输出作为输入进行转换
        return x #返回最终层的输出结果作为预测值

    def loss(self, x: np.ndarray, t: np.ndarray) -> float:
        y = self.predict(x) #执行前向传播，获取模型对输入x的预测结果y
        return self.lastLayer.forward(y, t) #调用预设的损失层（如SoftmaxWithLoss），计算预测值y与真实标签t之间的损失值并返回

#计算神经网络预测准确率。
    def accuracy(self, x: np.ndarray, t: np.ndarray) -> float:
        y = self.predict(x) #获取预测结果
        y_label = np.argmax(y, axis=1)#取概率最大索引作为预测标签
        if t.ndim == 2: # 若真实标签t为one-hot编码（二维），则同样转为索引形式，否则直接使用。
            t_label = np.argmax(t, axis=1)
        else:
            t_label = t
        return float(np.mean(y_label == t_label))

    def gradient(self, x: np.ndarray, t: np.ndarray) -> dict: #实现神经网络的反向传播计算梯度
        # forward
        self.loss(x, t)

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
        }
        return grads

    def numerical_gradients(self, x: np.ndarray, t: np.ndarray) -> dict: #计算两层神经网络各参数（W1/b1/W2/b2）的数值梯度。
        grads = {}

        def loss_W1():
            return self.loss(x, t)

        # 注意：数值梯度会直接改 params，所以要逐个算
        for key in ["W1", "b1", "W2", "b2"]:
            def f():
                return self.loss(x, t)
            grads[key] = numerical_gradient(f, self.params[key])

        return grads

# -------------------------
# Gradient Check (unit test)
# ------------------------- 
# 用于梯度检验，验证神经网络反向传播计算的梯度是否正确。


def gradient_check():
    np.random.seed(0) #固定随机种子确保可以复现
    # 初始化两层神经网络（4输入-5隐藏-3输出）
    net = TwoLayerNet(input_size=4, hidden_size=5, output_size=3, weight_init_std=0.01, seed=0)

    x = np.random.randn(2, 4)               # batch=2，生成2个4维随机输入样本
    t = np.array([0, 2])                    # label index (0..C-1)，对应样本的真实标签（0和1类）

    grad_backprop = net.gradient(x, t)      # 反向传播计算解析梯度
    grad_numerical = net.numerical_gradients(x, t) # 数值法计算近似梯度

     # 逐参数对比两种梯度计算结果
    for k in grad_backprop.keys():
        diff = np.mean(np.abs(grad_backprop[k] - grad_numerical[k]))  # 计算平均绝对误差
        print(f"{k}: mean abs diff = {diff:.8f}")                     # 输出误差值（理想应<1e-8）


if __name__ == "__main__":
    gradient_check()