# 深度学习-全连接层神经网络

## 第一次，纯梯度下降，没有加任何优化器(1-two_layer_net_min)

## 第二次，三种优化器（SGD/Momentum/Adam）之间的对比，发现momentum最后会有回落(2-two_layer_net_min)

## 第三次，加入了学习率衰减（step decay），momentum问题回落问题解决(3-two_layer_net_min)
- 但在SGD一直在0.46不变，主要是没有在每个优化器开始前置随机种子。
- 重要的实验规范：1.固定mini-batch序列，2.至少对每个方法重复多次取平均。

## 第四次，添加了随机种子重置，确保每个优化器使用相同的初始参数和数据顺序(3-two_layer_net_min)
- 更严格的方法是预先生成一个batches = [.....]列表，三种优化器都用同一份batches。
- 此时，现在的差异更能归因于优化器本身，而不是“刚好抽到了更容易的 batch”
- 参数确实在调整，可以看到损失函数确实在下降，但分类决策边界没有跨过关键阈值，所以 acc 不变。
  1. 学习率不合适。
  2. 现在的初始化尺度（0.01）导致早期信号偏弱。

## 第五次，只修改SGD的lr=0.5,他的效果高度依赖lr(3-two_layer_net_min)
- 通常SGD配合1.更仔细地lr选择；2.学习率衰减（step/cosine）；3.常常再加 Momentum（变成“SGD+Momentum”）

## 第六次初始化对比（0.01 vs Xavier vs He）(4-two_layer_net_min)
1. 给 TwoLayerNet 增加一个 init 参数（核心改动）
2. 新增一个对比初始化的函数（固定优化器，只换 init）
3. 改 main，让它只跑你想跑的实验
### 第六次结果
1.std=0.01明显起步慢，xavier和c3.he起步快很多，且he略优于xavier，
  - 说明初始化尺度决定了早期信号/梯度能不能顺畅传播。0.01 太小，会导致 logits 初期太接近 0、梯度偏弱，于是 SGD 要花很久才能“推开”决策边界。
  - 同时，toy太简单，所有差距不大。在更深网络/更难数据上，初始化差距会被放大。

1. 为什么 ReLU 理论上更适合 He，但你这里 Xavier/He 差不多？网络很浅（只有 1 个 ReLU 隐层）且任务很简单
  - He 的优势主要在“更深的 ReLU 网络”。
  - 即便 toy 数据简单，训练过程也会有波动。研究里常见的处理：加学习率衰减或者 early stopping（取验证集最优点）

## 第6次改动:实现 BatchNorm 层并插入网络(5-two_layer_net_min)
1. 在进行第六次改动之前,对之前的code进行小范围修改。
   - 在run_compare_optimizers()函数中最终打印缩进有问题,将final的统计缩进外层循环。
   - 学习率衰减不要对Adam/Momentum一刀切.修改
   - 在cross_entropy_error()和SoftmaxWithLoss.backward()中对标签形状不够鲁棒,将其强制拉平成1维会更稳
   - 在twolayernet函数中删除一个没有用处的函数loss_w1
   - 实现batchnorm的关键改造
     - BatchNorm/Dropout 都需要一个东西：训练/推理开关 train_flg
     - 在predict()函数中加入这个开关。
2. 实现BatchNorm层
   - 将accuracy()改为推理模式,不然后面加BN/Dropout,评估会抖.predict(x,train_flg = False)
   - 新增BatchNorm类,和 Affine/ReLU 放一起
   - 在twolayernet()的初始化中,新增gamma1/beta1参数
   - 在twolayernet()的初始化中,在layers中插入BatchNorm层
   - 在twolayernet()的gradient()中,收集gamma1/beta1的梯度
   - 修改在gradient_check()中的batch.
   - 把数值梯度的步长h调小一点。

## 第七次改动加了 BN 后,Momentum/Adam 反而变得更抖？怎么调参让它更稳？(6-two_layer_net_min)
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
  
# 深度学习-卷积神经网络

## CNN第一次(1-SimpleConvNet.py)
###  1.修改
  1. 先只写 forward + 缓存
  2. 添加pooling的forwoard
  3. 添加pooling的backward
  4. 添加conv的backward
  5. 添加一个flatten层
  6. 添加simpleconvnet类,test1
  7. 将解析度改为从score直接反转ones,暂时不走lastlayer.
  8. 把“抽查方式”改成更稳定的误差度量
### 2.结果
  1. (10, 16, 32, 32)
  2. (10, 16, 16, 16)
  3. (10, 16, 32, 32) 0.75
  4. y:(10, 16, 32, 32);dx: (10, 3, 32, 32);dW: (16, 3, 3, 3);db: (16,)
  5. (2, 1152) (2, 8, 12, 12)
  6. W1 partial check max abs diff: 0.000372369200343059
  7. W1(sum score) partial check max abs diff: 8.503611318383397e-05
  8. W1(sum score) partial check max relative error: 0.00010282018549844872  
     
## CNN第二次修改
- 主要是进行mnist数据的跑通，具体数据来源以及操作过程直接看2-simpleconvnet.py

