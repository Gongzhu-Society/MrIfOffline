Greedbatch是四个Greed对战的记录。
所采用的MrGreed是2020.7.10时调教好的，sample数量N取为20，大约能战胜 MrIf 90分的MrGreed。
parse_for_shi.py 是一个parse日志的demo，为那些不善于文字处理的同学准备的。

他们有微妙不同，如下。
乘二的意思是同样的牌局打两遍，乘四是打四遍。
打好几遍是因为一样的牌局MrGreed也因为sample的不同出不同的牌。
512x2: 1、2、3、4、5、6、7、8
256x4: 9、10、11、12