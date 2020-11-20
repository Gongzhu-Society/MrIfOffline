# MrIfOffline
Offline(testing) version of several AIs.

### 文件描述

* OfflineInterface.py: 线下游戏的接口, 用于对接不同的AI.
* Util.py            : Utility data and functions. Include
    * functions: `log, calc_score, cards_order, ...`
    * datas    : `ORDER_DICT, INIT_CARDS, SCORE_DICT, ...`
* MrRandom.py        : 最基本的机器人, 按 Ruichen Li 的接口定义了 `self.cards_list, self.history, self.cards_on_table, self.score` 等对象变量, 定义了几个 utility function, 之后的机器人都应该继承自 MrRandom. 同时给出了一个 Human 类表示人类, Human pick_a_card 时会请求人类输入.
* MrIf.py, MrGreed.py: 两个使用一些 if statements 来打牌的 AI, MrGreed 更强, 也是目前为止最强的.
* MrNN_Trainer.py    : 训练神经网络.
* MrNN.py            : 使用神经网络.

### MrNN
以下是两个MrNN对战两个MrIf 1024x2 盘的结果, 结果不尽如人意. 但是好在他很快, 出每张牌只需要大约 72/(1024x2)/26=1.3ms

```
20/07/15 22:56:57 890 [INFO,stat_ai:143] time consume: 72s
20/07/15 22:56:57 904 [WARN,stat_ai:152]  0+2 - 1+3: -50.79 5.72
```

对战MrRandom试试, 局数不变.

```
20/07/15 23:02:14 962 [INFO,stat_ai:144] time consume: 72s
20/07/15 23:02:14 983 [WARN,stat_ai:153]  0+2 - 1+3: 155.14 5.39
```

### MrGreed

MrGreed只看重眼前利益（这一圈的收益），不会用历史估计对手手牌（但是会考虑断章），也不会考虑自己之后怎么办。

MrGreed是用递归的方法定义的，首先，如果他是最后一个出牌的，他可以按照他的价值观（BURDEN_DICT）知道自己该怎么打。
如果他是倒数第二个则对可能的牌的分布采样，并假定最后一个人和自己打法一致，进而判断自己该怎么打。一直递归到第一个打的。

MrGreed不会再把猪塞给队友了。MrGreed会部分地考虑队友的感受，比如该它打牌，知道队友断某一门，自己又有这一门最小的，他就会打这一门好舒缓队友的压力。

下面是 MrGreed 在采样为10时对战 MrRandom 的战绩，对战局数为256x4

```
20/07/05 15:36:46 058 [INFO,stat_ai:137]  0+2 - 1+3: 272.01 7.48
```

下面是 MrGreed 在采样为10时对战 MrIf 的战绩，对战局数为256x4

```
20/07/05 15:40:35 939 [INFO,stat_ai:137]  0+2 - 1+3: 70.78 8.41
```

每次模拟挑选的可能数N是MrGreed的重要可调参数，理论上N越大MrGreed越强。
为了研究N对MrGreed强弱的具体影响，改变不同的 N 让 MrGreed 和 MrIf 对战，记录他们的分差

N     |5    |10   |20   |30   |40
------|:---:|:---:|:---:|:---:|:-:
vs If |63.6 |70.8 |80.8 |82.4 |太卡了，不可行
Sigma |8.7  |8.4  |8.5  |8.3  |/

发牌的抽样算法是速度的瓶颈，换用 发牌-交换 算法之后，对其强弱和速度测试如下。
发牌-交换 算法是先随机发牌直到发到合法的，再在这个基础上不断交换获得一系列合法发牌的算法。
可见“发牌-交换”算法让MrGreed变稳定了，但是变菜了，不过这个菜是可以用N的增加来弥补的。

N     |5    |10   |20   |30   |40   |50   |60
------|:---:|:---:|:---:|:---:|:---:|:---:|:-:
vs If |51.8 |50.2 |68.61|74.9 |76.1 |86.5 |78.2
Sigma |8.4  |8.3  |8.3  |8.4  |8.2  |8.0  |8.7
t(s)  |34   |62   |117  |178  |223  |285  |343

最早版本的MrGreed的价值观是不对的，现在做如下改进：1.C10的含义不再是-60而是真实的乘二；2.SA当猪已经打出就不在可怕。
于是他就能：1.在自己得了羊并且红桃和猪打得差不多时主动去抢变压器；2.知道对手的某一个人有大负分，就把变压器给他；3.知道队友有更多负分，就主动得变压器；
4.在猪已经出过的情况下不那么着急出手SA。理论上这会让他更强。我还更新了BURDEN_DICT，做了些调整。
分数提高并不多，可能是在细枝末节优化意义不大，也可能是我哪里代码写错了，毕竟这部分不好实现，写错了但能跑也有可能。

N     |5    |10   |20   |30
------|:---:|:---:|:---:|:---:
vs If |56.8 |60.0 |80.0 |86.2
Sigma |8.5  |8.0  |7.7  |8.0
t(s)  |44   |85   |166  |244

再加入一些人类经验：开始时打短的，结果明显变强了。

N     |5    |10   |20
------|:---:|:---:|:---:
vs If |57.7 |80.1 |96.5
Sigma |7.9  |8.1  |8.0

再改进：自己有猪时猪圈也淡然处之.没变强也没变弱吧。

N     |5    |10   |20
------|:---:|:---:|:---:
vs If |72.7 |81.1 |89.3
Sigma |7.9  |7.9  |7.7

### MrIf 和 MrRandom

MrRandom是按规则随即出牌的玩家，也是将来等级分规则的0分的基准

而 MrIf 是用 Ifs 判断几个最基础的规则其他情况随即出牌的玩家，他的规则包括：

    如果随便出
        从所剩张数少的花色开始，如果没有“危险牌”，就出这个花色大的
        尽量不出猪、猪圈、变压器、比变压器大的、红桃AK、羊
    如果是贴牌，按危险列表依次贴，没有危险列表了，贴短的
    如果是猪牌并且我的猪剩两张以上
        如果我有猪并且有人打过猪圈，贴猪
        如果我是最后一个并且前面没认出过猪，打除了猪之外最大的
        其他情况打不会得猪的
    如果是变压器并且草花剩两张以上
        类似于猪
    如果是羊并且剩两张以上
        如果我是最后一个，我有羊，并且前面的牌都比羊小，打羊
        其他情况打不是羊的最大的
    如果是红桃，尽可能躲，捡大的贴

下面是 MrIf 对战 MrRandom 的战绩，对战局数为256x16

```
20/07/04 22:35:24 511 [INFO,stat_random:137]  0+2 - 1+3: 194.00 3.63
```

More statistics for MrIf and MrRandom see Appendix A.

### Appendix A: MrRandom 和 MrIf 的详细统计信息
__第一个数字（比如下面第一行的-64.01）是平均每局得分，第二个（比如110.97）是得分的方差__

__详细统计信息可以用于纠错__

四个Mr. Random打1024x16局：
```
20/07/01 23:05:30 214 [INFO,stat_random:114] -64.01 110.97
20/07/01 23:05:30 217 [INFO,stat_random:114] -63.44 111.01
20/07/01 23:05:30 220 [INFO,stat_random:114] -62.29 109.91
20/07/01 23:05:30 223 [INFO,stat_random:114] -64.00 110.39
```

教会Mr. If先把短的花色打光再贴牌

    如果随便出
        从所剩张数少的花色开始，如果没有“危险牌”，就出这个花色大的
        尽量不出猪、猪圈、变压器、比变压器大的、红桃AK、羊
    如果是贴牌，按危险列表依次贴，没有危险列表了，贴短的
```
三个Mr. Random和一个Mr. If打128x16局：
20/07/02 12:30:33 370 [INFO,stat_random:116] -35.82 96.22
20/07/02 12:30:33 371 [INFO,stat_random:116] -75.61 117.02
20/07/02 12:30:33 372 [INFO,stat_random:116] -72.72 112.31
20/07/02 12:30:33 372 [INFO,stat_random:116] -73.02 112.53
两个Mr. Random和两个Mr. If打128x16局（邻）：
20/07/02 12:38:59 021 [INFO,stat_random:119] -42.78 95.74
20/07/02 12:38:59 022 [INFO,stat_random:119] -43.41 100.92
20/07/02 12:38:59 022 [INFO,stat_random:119] -90.96 125.14
20/07/02 12:38:59 023 [INFO,stat_random:119] -85.41 122.00
两个Mr. Random和两个Mr. If打128x16局（对）：
20/07/02 14:18:51 948 [INFO,stat_random:119] -42.74 99.39
20/07/02 14:18:51 949 [INFO,stat_random:119] -88.81 121.30
20/07/02 14:18:51 950 [INFO,stat_random:119] -44.93 97.94
20/07/02 14:18:51 950 [INFO,stat_random:119] -82.64 123.54
一个Mr. Random和三个Mr. If打128x16局：
20/07/02 12:39:33 774 [INFO,stat_random:119] -53.20 107.56
20/07/02 12:39:33 774 [INFO,stat_random:119] -56.47 106.12
20/07/02 12:39:33 775 [INFO,stat_random:119] -52.94 109.28
20/07/02 12:39:33 776 [INFO,stat_random:119] -102.85 131.01
四个Mr. If打128x16局：
20/07/02 12:40:30 030 [INFO,stat_random:119] -63.34 114.67
20/07/02 12:40:30 031 [INFO,stat_random:119] -67.39 121.14
20/07/02 12:40:30 031 [INFO,stat_random:119] -68.96 122.06
20/07/02 12:40:30 032 [INFO,stat_random:119] -68.15 114.92
```

教会Mr. If红桃草花黑桃方片的基本打法

    如果是猪牌并且我的猪剩两张以上
        如果我有猪并且有人打过猪圈，贴猪
        如果我是最后一个并且前面没认出过猪，打除了猪之外最大的
        其他情况打不会得猪的
    如果是变压器并且草花剩两张以上
        类似于猪
    如果是羊并且剩两张以上
        如果我是最后一个，我有羊，并且前面的牌都比羊小，打羊
        其他情况打不是羊的最大的
    如果是红桃，尽可能躲，捡大的贴
```
三个Mr. Random和一个Mr. If打128x16局：
20/07/02 14:15:45 410 [INFO,stat_random:119] -2.30 69.36
20/07/02 14:15:45 411 [INFO,stat_random:119] -83.91 118.71
20/07/02 14:15:45 411 [INFO,stat_random:119] -88.05 121.81
20/07/02 14:15:45 412 [INFO,stat_random:119] -91.43 123.46
两个Mr. Random和两个Mr. If打128x16局（邻）：
20/07/02 14:16:11 780 [INFO,stat_random:119] -18.26 80.08
20/07/02 14:16:11 780 [INFO,stat_random:119] -17.73 79.66
20/07/02 14:16:11 781 [INFO,stat_random:119] -117.96 138.31
20/07/02 14:16:11 782 [INFO,stat_random:119] -125.02 141.88
两个Mr. Random和两个Mr. If打128x16局（对）：
20/07/02 14:17:38 259 [INFO,stat_random:119] -22.49 81.92
20/07/02 14:17:38 259 [INFO,stat_random:119] -115.66 141.13
20/07/02 14:17:38 260 [INFO,stat_random:119] -19.26 77.73
20/07/02 14:17:38 261 [INFO,stat_random:119] -118.54 140.22
一个Mr. Random和三个Mr. If打128x16局：
20/07/02 14:16:33 123 [INFO,stat_random:119] -42.94 96.30
20/07/02 14:16:33 123 [INFO,stat_random:119] -37.52 101.65
20/07/02 14:16:33 124 [INFO,stat_random:119] -38.59 96.96
20/07/02 14:16:33 125 [INFO,stat_random:119] -169.94 167.55
四个Mr. If打128x16局：
20/07/02 14:16:53 298 [INFO,stat_random:119] -69.90 121.73
20/07/02 14:16:53 298 [INFO,stat_random:119] -66.86 120.47
20/07/02 14:16:53 299 [INFO,stat_random:119] -70.22 121.32
20/07/02 14:16:53 300 [INFO,stat_random:119] -69.22 121.62
```
