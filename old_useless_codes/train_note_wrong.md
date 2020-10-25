## 训练神经网络的笔记
## 由于把训练集当测试集了, 所以结论大部分都是错的

### 2020.7.13
(0) 使用8层神经网络模仿MrGreed打第一张牌的行为最高模仿率大约是48%

### 2020.7.14
(1) 转而使用神经网络模仿MrGreed打最后一张牌的行为, MrGreed打最后一张牌的逻辑很简单,一定好学. 并且果然取得了突破, 使用
```
NN_Last(
  (fc1): Linear(in_features=260, out_features=416, bias=True)
  (fc2): Linear(in_features=416, out_features=208, bias=True)
  (fc3): Linear(in_features=208, out_features=104, bias=True)
  (fc4): Linear(in_features=104, out_features=52, bias=True)
  (fc5): Linear(in_features=52, out_features=52, bias=True)
  (fc6): Linear(in_features=52, out_features=52, bias=True)
) 228020(总参数个数)
```
输入参数是我手里的牌(52), 另外三个人总共手里有的牌(52), 前三张牌(52x3). 使用batch2和batch3的9949+9970个数据进行训练, 使用batch1的前128墩的1255个数据测试, 经过585个epoch后, (训练集loss, 测试集loss, 测试集模仿率) 达到了: 0.012297 0.625901 0.962025. 96.2%是非常高的正确率, 与之相对的, 随机初始化的神经网络准确率大约是 26%-29%.

(2) 但是转念一想, MrGreed打最后一张牌的逻辑也很简单, 只有几个if, 所以神经网络拟合的好不足为奇, 同时这说明我应当可以缩减神经网络参数的个数达到一样好的效果. 慢慢小心翼翼地缩小神经网络每一层的大小和层数, 保持训练集和测试集不变, 得到如附录一所示结果. 从附录一结果可见, 当从7层网络模型68万个参数约减神经网络, 一直到3层3.5万个参数, 其表现都是在变好的, 不但模仿率维持在98%, 测试集loss更是从0.6下降到了0.12, 我认为这是减弱了过拟合的效果. 附录一中效果最好的结构为
```
NN_Last(
  (fc1): Linear(in_features=260, out_features=156, bias=True)
  (fc3): Linear(in_features=156, out_features=52, bias=True)
  (fc6): Linear(in_features=52, out_features=52, bias=True)
  (fc7): Linear(in_features=52, out_features=52, bias=True)
) 54392 500: 0.014685 0.136816 0.984461
```
再继续缩减神经网络, 其表现慢慢变差, 但是没有变差太多, 一直到1层1.3万个参数模仿率下降到了91%, 测试集loss上升到了0.35.

(3) 又想到这个MrNN将来有一天要接入蒙特卡洛树搜索自己强化自己的, 所以要赋予这个神经网络足够的潜能, 为此将四个人手中有分的牌也加入输入中, 也把四个人的断门输入神经网络, 这为输入增加了16x4+4x4个. 使用如下的神经网络
```
NN_Last(
  (fc1): Linear(in_features=340, out_features=256, bias=True)
  (fc2): Linear(in_features=256, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=64, bias=True)
  (fc4): Linear(in_features=64, out_features=64, bias=True)
  (fc5): Linear(in_features=64, out_features=52, bias=True)
) 135988(总参数个数)
```
输入参数比之前的增加了80个, 是四个人手里的分和他们的断门. 仍然使用之前的测试集, 经过500个epoch之后, (训练集loss, 测试集loss, 测试集模仿率) 达到了: 0.003153 0.477918 0.978062. 比(2)中的最好表现变差了, 似乎提供的信息太多把神经网络绕晕了, 让他抓不到重点, 但是想到它的潜力和也很高的正确率, 就这样吧. 接下来要看看倒数第二个出牌的能训练到怎么样了.

(4) 仍然使用batch1和batch2的10106+10035个数据训练, 用batch1的一部分一共1233个测试. 无论是5层6层还是7层网络都严重过拟合, 6层126836个参数的网络最后结果为0.021057 2.983818 0.898305(600 epoch), 6层139188参数的网络最后结果为0.011506 2.793123 0.891041(600 epoch),如下的7层网络更过分.
```
NN_Third(
  (fc1): Linear(in_features=288, out_features=256, bias=True)
  (fc2): Linear(in_features=256, out_features=128, bias=True)
  (fc6): Linear(in_features=128, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=64, bias=True)
  (fc7): Linear(in_features=64, out_features=64, bias=True)
  (fc4): Linear(in_features=64, out_features=64, bias=True)
  (fc5): Linear(in_features=64, out_features=52, bias=True)
) 143348 600: 0.019255 4.205298 0.878935 (严重过拟合)
```
可见过拟合现象严重, 增加batch4的10122个训练数据, 并缩减神经网络至7层, 获得了不错的结果, 再增加batch5, 6效果也不会变差(曾经一度用更多的数据训练效果反而会变差, 我不知道为什么也无法复现).
```
NNN_Third(
  (fc1): Linear(in_features=288, out_features=256, bias=True)
  (fc2): Linear(in_features=256, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=64, bias=True)
  (fc4): Linear(in_features=64, out_features=64, bias=True)
  (fc5): Linear(in_features=64, out_features=64, bias=True)
  (fc7): Linear(in_features=64, out_features=52, bias=True)
) 126836 600: 0.009652 0.010737 0.992701
```
训练再深一层的神经网络效果也很好, 我觉得Third比Last深两层是很合理的. 用3个, 4个, 5个文件的数据训练均能良好收敛.
```
NN_Third(
  (fc1): Linear(in_features=288, out_features=256, bias=True)
  (fc2): Linear(in_features=256, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=64, bias=True)
  (fc4): Linear(in_features=64, out_features=64, bias=True)
  (fc5): Linear(in_features=64, out_features=64, bias=True)
  (fc6): Linear(in_features=64, out_features=64, bias=True)
  (fc7): Linear(in_features=64, out_features=52, bias=True)
) 130996 600: 0.011438 0.011930 0.992701
```

### 2020.7.15
(5) 现开始训练第二个打牌的神经网络, 先使用和Third相同的结构, 结果如下.
```
NN_Second [(236, 256), (256, 128), (128, 64), (64, 64), (64, 64), (64, 64), (64, 52)] 117684
600: 0.021357 0.025694 0.984528
```
按我的偏见加深两层, 结果也不错
```
NN_Second [(236, 256), (256, 128), (128, 64), (64, 64), (64, 64), (64, 64), (64, 64), (64, 64), (64, 52)] 126004
600: 0.036770 0.035578 0.982085
```
同样的, 采用上面第二个打牌的神经网络作为第一个打牌的训练, 结果如下.
```
NN_First [(184, 256), (256, 128), (128, 64), (64, 64), (64, 64), (64, 64), (64, 64), (64, 64), (64, 52)] 112692
600: 0.089018 0.086637 0.966146
```
再加深两层, 训练似乎遇到了困难(300: 1.681613 1.697264 0.349609), 增加一些训练集, 用六个文件训练, 就解决了
```
NN_First [(184, 256), (256, 128), (128, 64), (64, 64), (64, 64), (64, 64), (64, 64), (64, 64), (64, 64), (64, 64), (64, 52)] 121012
600: 0.093378 0.072306 0.967448
```
(6) (3)中说到, 增加了输入和深度之后,训练效果下降了, 但是同等地增加训练集, 将batch4加入进去, 正确率竟然达到了惊人的100%!
```
NN_Last [(340, 256), (256, 128), (128, 64), (64, 64), (64, 52)] 135988
100: 0.003760 0.001901 1.000000
```

### 附录一: 不断缩小Last的网络大小观察其表现
```
NN_Last(
  (fc1): Linear(in_features=260, out_features=832, bias=True)
  (fc2): Linear(in_features=832, out_features=416, bias=True)
  (fc3): Linear(in_features=416, out_features=208, bias=True)
  (fc4): Linear(in_features=208, out_features=104, bias=True)
  (fc5): Linear(in_features=104, out_features=52, bias=True)
  (fc6): Linear(in_features=52, out_features=52, bias=True)
  (fc7): Linear(in_features=52, out_features=52, bias=True)
) 683124 1250: 0.016590 1.260692 0.978062
NN_Last(
  (fc1): Linear(in_features=260, out_features=416, bias=True)
  (fc2): Linear(in_features=416, out_features=208, bias=True)
  (fc3): Linear(in_features=208, out_features=104, bias=True)
  (fc4): Linear(in_features=104, out_features=52, bias=True)
  (fc5): Linear(in_features=52, out_features=52, bias=True)
  (fc6): Linear(in_features=52, out_features=52, bias=True)
) 228020 585: 0.012297 0.625901 0.962025
NN_Last(
  (fc1): Linear(in_features=260, out_features=260, bias=True)
  (fc3): Linear(in_features=260, out_features=208, bias=True)
  (fc4): Linear(in_features=208, out_features=104, bias=True)
  (fc5): Linear(in_features=104, out_features=52, bias=True)
  (fc6): Linear(in_features=52, out_features=52, bias=True)
  (fc7): Linear(in_features=52, out_features=52, bias=True)
) 154856 550: 0.015294 0.560318 0.978062
NN_Last(
  (fc1): Linear(in_features=260, out_features=156, bias=True)
  (fc3): Linear(in_features=156, out_features=104, bias=True)
  (fc5): Linear(in_features=104, out_features=52, bias=True)
  (fc6): Linear(in_features=52, out_features=52, bias=True)
  (fc7): Linear(in_features=52, out_features=52, bias=True)
) 68016 500: 0.007516 0.371668 0.979890
NN_Last(
  (fc1): Linear(in_features=260, out_features=156, bias=True)
  (fc3): Linear(in_features=156, out_features=52, bias=True)
  (fc6): Linear(in_features=52, out_features=52, bias=True)
  (fc7): Linear(in_features=52, out_features=52, bias=True)
) 54392 500: 0.014685 0.136816 0.984461
NN_Last(
  (fc1): Linear(in_features=260, out_features=104, bias=True)
  (fc3): Linear(in_features=104, out_features=52, bias=True)
  (fc6): Linear(in_features=52, out_features=52, bias=True)
  (fc7): Linear(in_features=52, out_features=52, bias=True)
) 38116 500: 0.017906 0.138936 0.987203
NN_Last(
  (fc1): Linear(in_features=260, out_features=104, bias=True)
  (fc3): Linear(in_features=104, out_features=52, bias=True)
  (fc7): Linear(in_features=52, out_features=52, bias=True)
) 35360 500: 0.084769 0.121791 0.972578
NN_Last(
  (fc1): Linear(in_features=260, out_features=52, bias=True)
  (fc3): Linear(in_features=52, out_features=52, bias=True)
  (fc7): Linear(in_features=52, out_features=52, bias=True)
) 19084 500: 0.108605 0.133050 0.965265
NN_Last(
  (fc1): Linear(in_features=260, out_features=52, bias=True)
  (fc7): Linear(in_features=52, out_features=52, bias=True)
) 16328 500: 0.162771 0.155999 0.957038
NN_Last(
  (fc1): Linear(in_features=260, out_features=52, bias=True)
) 13572 500: 0.368931 0.354754 0.913163
```
