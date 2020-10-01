# digix2020_search_rank1
华为digix算法大赛2020机器学习赛道-搜索相关性初赛A/B榜rank1

队名：忑廊埔大战拗芭码

分数：初赛A榜0.445196/B榜0.448975

排名：初赛A榜rank1/B榜rank1

项目的blog分享[链接](https://blog.csdn.net/weixin_40174982/article/details/108880726)

希望大家都能进入决赛，11月份南京见！

## 项目环境

Python 3.8

lightgbm

sklearn

pandas

numpy

tqdm

## 处理流程

在search下创建data文件夹，并将训练集、测试集A、测试集B的csv文件放在search/data/

运行reduce/reduce.py进行数据压缩

运行lgb_fold.py进行模型一的训练和推理

运行lgb_rank_fold.py进行模型二的训练和推理

运行result/fusion.py得到两个模型结果的融合

result文件夹中可得到最终结果文件submission.csv
