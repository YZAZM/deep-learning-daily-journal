from collections import OrderedDict #导入 OrderedDict，用于保持层的顺序
import numpy as np

#--------------------------
#读取数据,数据来源：https://www.kaggle.com/datasets/hojjatk/mnist-dataset
#--------------------------
def load_mnist_idx(train_images_path, train_labels_path, test_images_path, test_labels_path,
                   normalize=True, flatten=False, one_hot_label=False):
    #读取数据
    x_train = _load_idx3_ubyte(train_images_path)
    t_train = _load_idx1_ubyte(train_labels_path)
    x_test  = _load_idx3_ubyte(test_images_path)
    t_test  = _load_idx1_ubyte(test_labels_path)

    if normalize:
        x_train = x_train.astype(np.float32) / 255.0
        x_test  = x_test.astype(np.float32) / 255.0

    # 你的 CNN 代码一般用 NCHW: (N, C, H, W)
    if not flatten:
        x_train = x_train.reshape(-1, 1, 28, 28)
        x_test  = x_test.reshape(-1, 1, 28, 28)
    else:
        x_train = x_train.reshape(-1, 784)
        x_test  = x_test.reshape(-1, 784)

    if one_hot_label:
        t_train = _to_one_hot(t_train, 10)
        t_test  = _to_one_hot(t_test, 10)

    return (x_train, t_train), (x_test, t_test)

def _load_idx3_ubyte(path):#二进制打开图像文件
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28)

def _load_idx1_ubyte(path):
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data.astype(np.int64)

def _to_one_hot(t, num_classes):
    out = np.zeros((t.size, num_classes), dtype=np.float32)
    out[np.arange(t.size), t] = 1.0
    return out



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

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    input_data: (N, C, H, W)
    return: col (N*out_h*out_w, C*filter_h*filter_w)
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h) // stride + 1
    out_w = (W + 2*pad - filter_w) // stride + 1

    img = np.pad(input_data,
                 [(0,0), (0,0), (pad,pad), (pad,pad)],
                 mode="constant")

    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h) // stride + 1
    out_w = (W + 2*pad - filter_w) // stride + 1

    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)
    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H+pad, pad:W+pad]


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

class Flatten:
    def __init__(self):
        self.orig_shape = None

    def forward(self, x):
        self.orig_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.orig_shape)


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


#---------------------------
#卷积层类
#--------------------------
class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        """
        W: (FN, C, FH, FW)
        b: (FN,)
        """
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        # cache for backward
        self.x = None
        self.col = None
        self.col_W = None

        # grads
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, Cx, H, W = x.shape
        assert Cx == C, "Input channel must match filter channel"

        out_h = (H + 2*self.pad - FH) // self.stride + 1
        out_w = (W + 2*self.pad - FW) // self.stride + 1

        col = im2col(x, FH, FW, self.stride, self.pad)          # (N*out_h*out_w, C*FH*FW)
        col_W = self.W.reshape(FN, -1).T                        # (C*FH*FW, FN)

        out = np.dot(col, col_W) + self.b                       # (N*out_h*out_w, FN)
        out = out.reshape(N, out_h, out_w, FN).transpose(0, 3, 1, 2)  # (N, FN, out_h, out_w)

        self.x = x
        self.col = col
        self.col_W = col_W
        return out
    
    def backward(self, dout):
        FN, C, FH, FW = self.W.shape

        # (N, FN, out_h, out_w) -> (N*out_h*out_w, FN)
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        # db
        self.db = np.sum(dout, axis=0)

        # dW: (C*FH*FW, FN) -> (FN, C, FH, FW)
        dW = np.dot(self.col.T, dout)
        self.dW = dW.transpose(1, 0).reshape(FN, C, FH, FW)

        # dx: (N*out_h*out_w, C*FH*FW) -> col2im -> (N, C, H, W)
        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


#---------------------------
#池化层类
#--------------------------
class Pooling:
    def __init__(self, pool_h, pool_w, stride=2, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        # cache for backward
        self.x = None
        self.arg_max = None
        self.col = None
        self.col_arg = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = (H + 2*self.pad - self.pool_h) // self.stride + 1
        out_w = (W + 2*self.pad - self.pool_w) // self.stride + 1

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        # col shape: (N*out_h*out_w, C*pool_h*pool_w)
        col = col.reshape(-1, C, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=2)     # (N*out_h*out_w, C)
        out = np.max(col, axis=2)            # (N*out_h*out_w, C)

        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max
        self.col = col
        return out
    
    def backward(self, dout):
        # dout: (N, C, out_h, out_w)
        N, C, out_h, out_w = dout.shape

        # 展平成 (N*out_h*out_w, C)
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, C)
        # 构造 dmax: (N*out_h*out_w, C, pool_h*pool_w)
        dmax = np.zeros((dout.shape[0], C, self.pool_h * self.pool_w))
        # 把 dout 放回 argmax 的位置
        idx = self.arg_max.reshape(-1, C)
        for i in range(dout.shape[0]):
            dmax[i, np.arange(C), idx[i]] = dout[i]

        # 还原成 im2col 之前的列结构
        dcol = dmax.reshape(dout.shape[0], -1)  # (N*out_h*out_w, C*pool_h*pool_w)

        # col2im: 把列还原回 (N, C, H, W)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        return dx


#---------------------------
#SimpleConvNet类
#--------------------------
class SimpleConvNet:
    def __init__(self,
                 input_dim=(1, 28, 28),
                 conv_param={"filter_num": 8, "filter_size": 5, "pad": 0, "stride": 1},
                 hidden_size=10,  # 这里我们直接做 10 类输出，所以 hidden_size 可不用
                 output_size=10,
                 weight_init_std=0.01):

        C, H, W = input_dim
        FN = conv_param["filter_num"]
        FH = conv_param["filter_size"]
        FW = conv_param["filter_size"]
        pad = conv_param["pad"]
        stride = conv_param["stride"]

        # --- 1) 计算卷积输出尺寸 ---
        out_h = (H + 2*pad - FH) // stride + 1
        out_w = (W + 2*pad - FW) // stride + 1

        # --- 2) 计算池化输出尺寸（2x2, stride=2）---
        pool_h, pool_w, pool_stride, pool_pad = 2, 2, 2, 0
        pool_out_h = (out_h + 2*pool_pad - pool_h) // pool_stride + 1
        pool_out_w = (out_w + 2*pool_pad - pool_w) // pool_stride + 1

        # --- 3) Flatten 后的维度 ---
        flat_dim = FN * pool_out_h * pool_out_w  # TODO: 这里你可以 print 检查是否等于 1152

        # --- 4) 初始化参数 ---
        self.params = {}
        # W1: (FN, C, FH, FW)
        self.params["W1"] = weight_init_std * np.random.randn(FN, C, FH, FW)  # TODO: 确认形状
        self.params["b1"] = np.zeros(FN)

        # W2: (flat_dim, output_size)
        self.params["W2"] = weight_init_std * np.random.randn(flat_dim, output_size)  # TODO: 确认形状
        self.params["b2"] = np.zeros(output_size)

        # --- 5) 组装层 ---
        self.layers = OrderedDict()
        self.layers["Conv1"] = Convolution(self.params["W1"], self.params["b1"],
                                           stride=stride, pad=pad)
        self.layers["ReLU1"] = ReLU()
        self.layers["Pool1"] = Pooling(pool_h, pool_w, stride=pool_stride, pad=pool_pad)
        self.layers["Flatten"] = Flatten()
        self.layers["Affine1"] = Affine(self.params["W2"], self.params["b2"])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def loss_sumscore(self, x):
        score = self.predict(x)
        return np.sum(score)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y_label = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        return np.mean(y_label == t)
    
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward from last layer
        dout = 1
        dout = self.lastLayer.backward(dout)

        # backward through layers in reverse order
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # grads
        grads = {}
        grads["W1"] = self.layers["Conv1"].dW
        grads["b1"] = self.layers["Conv1"].db
        grads["W2"] = self.layers["Affine1"].dW
        grads["b2"] = self.layers["Affine1"].db
        return grads
    

# -------------------------
# Optimizers
# -------------------------
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
        lr_t = self.lr * (np.sqrt(1.0 - self.beta2 ** self.iter) /
                          (1.0 - self.beta1 ** self.iter))

        for k in params:
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grads[k]
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (grads[k] ** 2)
            params[k] -= lr_t * self.m[k] / (np.sqrt(self.v[k]) + self.eps)



if __name__ == "__main__":
    np.random.seed(0)  # 固定随机种子：每次运行可复现
    #注意数据和代码放在同一个文件下
    (train_x, train_t), (test_x, test_t) = load_mnist_idx(
        r"2.sleflearn/2-2.CNN/1.mnist/train-images-idx3-ubyte/train-images-idx3-ubyte",
        r"2.sleflearn/2-2.CNN/1.mnist/train-labels-idx1-ubyte/train-labels-idx1-ubyte",
        r"2.sleflearn/2-2.CNN/1.mnist/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte",
        r"2.sleflearn/2-2.CNN/1.mnist/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte",
        normalize=True,
        flatten=False,
        one_hot_label=True
    )


    print("train:", train_x.shape, train_t.shape, train_x.dtype, train_t.dtype)
    print("test :", test_x.shape, test_t.shape)

    # 1) 建网
    net = SimpleConvNet(
        input_dim=(1, 28, 28),
        conv_param={"filter_num": 8, "filter_size": 5, "pad": 0, "stride": 1},
        output_size=10,
        weight_init_std=0.01
    )

    # 2) 选优化器（先用 Adam，最容易跑起来）
    optimizer = Adam(lr=0.001)   # 先用 0.001；等你稳定后再尝试 0.0005 / 0.002

    # 3) 训练超参
    iters = 2000
    batch_size = 100
    eval_interval = 100

    train_size = train_x.shape[0]
    best_test_acc = 0.0
    best_params = None

    for it in range(iters):
        #1.学习率衰减，step decay,前期用大 lr 快速学，后期用小 lr 精修，更稳更高。
        if it == 1200:
            optimizer.lr *= 0.3 #每次衰减0.3倍
            print(">>> lr =", optimizer.lr)

        batch_mask = np.random.choice(train_size, batch_size, replace=False)
        x_batch = train_x[batch_mask]
        t_batch = train_t[batch_mask]

        grads = net.gradient(x_batch, t_batch)
        optimizer.update(net.params, grads)

        # 每隔 eval_interval 评估一次（评估用小子集提速）
        if it % eval_interval == 0:
            loss = net.loss(x_batch, t_batch)

            train_acc = net.accuracy(train_x[:1000], train_t[:1000])
            test_acc  = net.accuracy(test_x, test_t)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_params = {k: v.copy() for k, v in net.params.items()}

            print(f"iter {it:04d} | loss {loss:.4f} | train_acc {train_acc:.3f} | test_acc {test_acc:.3f} | best {best_test_acc:.3f}")

    # 最终用全量测试集跑一次（可能慢一点）
    final_test_acc = net.accuracy(test_x, test_t)
    print(f"final test acc: {final_test_acc:.4f}")

    if best_params is not None:
        for k in net.params:
            net.params[k][...] = best_params[k]
    best_test_acc_full = net.accuracy(test_x, test_t)
    print(f"best test acc (logged): {best_test_acc:.4f}")
    print(f"best params full test acc: {best_test_acc_full:.4f}")




