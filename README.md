# MrIfOffline
Offline(testing) version of several AIs.

### 文件描述

* __MrRandom.py__        : 最基本的机器人, 按 Ruichen Li 的接口定义了 `self.cards_list, self.history, self.cards_on_table, self.score` 等对象变量, 定义了几个 utility function, 之后的机器人都应该继承自 MrRandom. 同时给出了一个 Human 类表示人类, Human pick_a_card 时会请求人类输入.
* __MrIf.py, MrGreed.py__: 两个使用一些 if statements 来打牌的 AI, MrGreed 更强, 也是目前为止最强的.
* __MrRandTree.py__      : MCTS AI using random rollout policy. MrRandTree and MrGreed are the most powerful AI so far.
* __ScenarioGen.py__     : 生成可能的手牌情形(Scenario).
* __OfflineInterface.py__: 线下游戏的接口, 用于对接不同的AI.
* __Util.py__            : Utility data and functions. Including
    * functions: `log, calc_score, cards_order, ...`
    * datas    : `ORDER_DICT, INIT_CARDS, SCORE_DICT, ...`

### Git Multi-branch

* show git branches: `git branch`, the one with asterisk is the current branch.
* create new branch: `git branch branchname`
* switch to branch: `git checkout branchname`
* create new branch b2 on github based on local branch b1: `git push -u origin b1:b2`, b1 and b2 can have the same name

### MrZeroTree

* setup: 
```
git clone https://github.com/Gongzhu-Society/MrIfOffline.git
cd ./MrIfOffline
git clone https://github.com/Gongzhu-Society/ScenarioGenerator.git
git clone https://github.com/Gongzhu-Society/MCTS.git
./Manually_test.py
```
* Direct benchmark: `benchmark` in Manually_test.py
* Read a standard shuffle: `read_std_hands` in OfflineInterface.py
* Play one (or multiple) game: `play_a_test` in OfflibeInterface.py
* Train in single threading: MrZ_TB.py
* Train with multi-processing: MrZ_Trainer.py


### MrRandTree

MrRandTree is an MCTS AI with random rollout policy. The table below contains benchmarks for MrRandTree. Its scenario number is set to 5, iteration number to 200. MrIf's stats are from stats in Appendix A. One can see that MrRandTree is rather strong.

|Item  |MrIf v.s. MrRandom|MrRandTree v.s. MrRandom|MrRandTree v.s. MrIf|MrRandTree v.s. MrGreed|
|:----:|:----------------:|:----------------------:|:------------------:|:---------------------:|
|Mode  |1 v.s. 3          |1 v.s. 3                |1 v.s. 3            |2 v.s. 2               |
|Result|-2.3/87.8         |22.7/-88.2              |-29.4/-56.3         |-90.8/-92.3            |
|Sigma |1.5/2.7           |13.2/15.1               |20.3/13.6           |11.8/9.3               |
|Time  |                  |981s(16x4)              |1023s(16x4)         |7847s(128x2)           |

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

Fix $N_{sample}$ to 20, let MrGreed play against MrRandom and MrIf in 2 v.s. 2 mode, the results are as following table. Number of play is set to 1024x2

Opponent       |MrRandom|MrIf |
:-------------:|:------:|:---:|
Points(0+2-1-3)|275.21  |80.28|
Sigma          |5.28    |5.71 |
Time           |167s    |187s |

We can see that, by using new sampling package, our MrGreed gets faster(see the Table above above).

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
