## 训练神经网络的笔记

### 2020.7.13
使用8层神经网络模仿MrGreed打第一张牌的行为最高模仿率大约是48%

### 2020.7.14
1. 转而使用神经网络模仿MrGreed打最后一张牌的行为, MrGreed打最后一张牌的逻辑很简单,一定好学. 并且果然取得了突破, 使用
NN_Last(
  (fc1): Linear(in_features=260, out_features=416, bias=True)
  (fc2): Linear(in_features=416, out_features=208, bias=True)
  (fc3): Linear(in_features=208, out_features=104, bias=True)
  (fc4): Linear(in_features=104, out_features=52, bias=True)
  (fc5): Linear(in_features=52, out_features=52, bias=True)
  (fc6): Linear(in_features=52, out_features=52, bias=True)
) 228020(总参数个数)
经过585个epoch后: 0.012297 0.625901 0.962025(训练集loss,测试集loss,测试集模仿率). 96.2%是非常高的正确率, 与之相对的, 随机初始化的神经网络准确率大约是 26%-29%.

2. 但是转念一想, MrGreed打最后一张牌的逻辑也很简单, 只有几个if, 所以神经网络拟合的好不足为奇, 同时这说明我应当可以缩减神经网络参数的个数达到一样好的效果. 
慢慢小心翼翼地缩小神经网络每一层的大小和层数, 得到如附录一所示结果. 从附录一结果可见, 当从7层网络模型68万个参数约减神经网络, 一直到3层3.5万个参数, 其表现都是在变好的, 不但模仿率维持在98%, 测试集loss更是从0.6下降到了0.12, 我认为这是减弱了过拟合的效果. 再继续缩减神经网络, 其表现慢慢变差, 但是没有变差太多, 一直到1层1.3万个参数模仿率下降到了91%, 测试集loss上升到了0.35.

3.

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