# MrIfOffline
Offline(testing) version of several AIs.

### 文件描述

* __OfflineInterface.py__: 线下游戏的接口, 用于对接不同的AI.
* __Util.py__            : Utility data and functions. Include
    * functions: `log, calc_score, cards_order, ...`
    * datas    : `ORDER_DICT, INIT_CARDS, SCORE_DICT, ...`
* __MrRandom.py__        : 最基本的机器人, 按 Ruichen Li 的接口定义了 `self.cards_list, self.history, self.cards_on_table, self.score` 等对象变量, 定义了几个 utility function, 之后的机器人都应该继承自 MrRandom. 同时给出了一个 Human 类表示人类, Human pick_a_card 时会请求人类输入.
* __MrIf.py, MrGreed.py__: 两个使用一些 if statements 来打牌的 AI, MrGreed 更强, 也是目前为止最强的.
* __ScenarioGen.py__: 生成可能的手牌情形(Scenario).
* MrNN_Trainer.py    : 训练神经网络.
* MrNN.py            : 使用神经网络.

### ScenarioGen

生成可能的手牌情形(Scenario). 用法如

```
sce_gen=ScenarioGen(self.place,self.history,self.cards_on_table,self.cards_list,number=20,method=None)
for cards_list_list in sce_gen:
    print(cards_list_list) #will get things like [['C4', 'C2', 'C6'], ['SQ', 'D8'], ['D2', 'DJ']]
```

其中`number`是要sample的数量; `method=None` 表示让程序自动选择 sample 方法, 程序会自动从 shot and test 和 constryct by table 中选一个, 如果选择了 constryct by table 并且可能的所有情况数小于想要的 sample 数, 则返回所有情况. 如果想指定方法请自行阅读代码或咨询作者.

现在进行 benchmark, 使用两个 MrGreed 对战两个 MrIf 256x2 局, sample number 设为20, cpu 为 Intel i9-9960X, METHOD1_PREFERENCE=0. 结果如下

Item   |Using Shot and Test|Full ScenarioGen|
:-----:|:-----------------:|:--------------:|
Time(s)|49                 |47              |

平均每个 sample 需要`48/(256*2*26*20)=180us`.

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

MrGreed不会再把猪塞给队友了。MrGreed会部分地考虑队友的感受，比如该它打牌，知道队友断某一门，自己又有这一门最小的，他就会打这一门好舒缓队友的压力. 他还知道自己第一个出牌时应该先打短的.

下表是 MrGreed 对战 Mrif 的战绩, 对战局数为 1024x2, $N_{sample}$ 是 MrGreed 的重要可调参数, 它是 MrGreed 打牌时 sample 的个数, 理论上越大越强.

$N_{sample}$    |5             |10            |20            |40
:--------------:|:------------:|:------------:|:------------:|:------------:
Points over MrIf|$71.82\pm5.76$|$75.41\pm5.55$|$83.73\pm5.62$|$86.44\pm5.63$
Time Comsumed   |61s           |114s          |218s          |419s

### MrIf 和 MrRandom

MrRandom是按规则随机出牌的玩家，也是将来等级分规则的0分的基准

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

MrIf 对战 MrRandom, 对战局数为 256x16 时, `0+2 - 1+3: 194.00 3.63`. More statistics for MrIf and MrRandom see Appendix A.

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
四个Mr. If打128x16局：
20/07/02 14:16:53 298 [INFO,stat_random:119] -69.90 121.73
20/07/02 14:16:53 298 [INFO,stat_random:119] -66.86 120.47
20/07/02 14:16:53 299 [INFO,stat_random:119] -70.22 121.32
20/07/02 14:16:53 300 [INFO,stat_random:119] -69.22 121.62
```
