# MrIfOffline
An offline(testing) version of Mr. If 
线下版本的茹先生（MrIf）

### AI介绍
文件中的MrRandom是按规则随即出牌的玩家，也是将来等级分规则的0分的基准

而MrIf是用Ifs判断几个最基础的规则其他情况随即出牌的玩家，他的规则包括：

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

### MrIf 对 MrRandom

```
20/07/04 22:35:24 505 [INFO,stat_random:131] 0th player: -20.81 1.22
20/07/04 22:35:24 506 [INFO,stat_random:131] 1th player: -119.55 2.14
20/07/04 22:35:24 507 [INFO,stat_random:131] 2th player: -21.22 1.22
20/07/04 22:35:24 507 [INFO,stat_random:131] 3th player: -116.48 2.19
20/07/04 22:35:24 509 [INFO,stat_random:133] 0 2 player: -42.03 1.59
20/07/04 22:35:24 510 [INFO,stat_random:135] 1 3 player: -236.03 2.26
20/07/04 22:35:24 511 [INFO,stat_random:137]  0+2 - 1+3: 194.00 3.63
```

### MrRandom 和 MrIf 的详细统计信息
#### 第一个数字（比如下面第一行的-64.01）是平均每局得分，第二个（比如110.97）是得分的方差
#### 详细统计信息可以用于纠错

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
