
2018.9.10 更新:
- 参考OpenNRE使用mask可以快速计算piece wise pooling.
    - 修改NYT 53类数据处理 (完成): 将dataset目录下的extract.cpp 替换原始数据extract_cpp目录下的extract.cpp
    - 修改NYT 27类数据处理 (未完成)
    

使用Pytorch 复现 PCNN+MIL (Zeng 2015) 与 PCNN+ATT (Lin 2016), 以及两个模型在两个大小版本的数据集上(27类关系/53类关系)的表现对比。



相关博客:

- [关系抽取论文笔记](http://shomy.top/2018/02/28/relation-extraction/)

- [复现结果说明](http://shomy.top/2018/07/05/pytorch-relation-extraction/)



在代码的组织,结构设计上,  主要参考 [陈云Pytorch实战指南](https://zhuanlan.zhihu.com/p/29024978) (个人推荐)。因此一些实现细节就不再赘述了，可以参考陈云的实战指南。



## 实现总览


环境:

- Python 2.X
- Pytorch 0.3.1
- fire

简单介绍主要目录：

```
├── checkpoints         # 保存预加载模型
├── config.py             # 参数
├── dataset                # 数据目录
│ ├── FilterNYT         # SMALL 数据
│ ├── NYT                 # LARGE 数据
│ ├── filternyt.py
│ ├── __init__.py
│ ├── nyt.py
├── main_mil.py       # PCNN+ONE 主文件
├── main_att.py        # PCNN+ATT 主文件
├── models               # 模型目录
│ ├── BasicModule.py
│ ├── __init__.py
│ ├── PCNN_ATT.py
│ ├── PCNN_ONE.py
├── plot.ipynb
├── README.md
├── utils.py                # 工具函数
```



这份代码基本上是按照陈云的指南模仿来写的。 数据模型分开，参数/配置单独文件， 并且使用fire 库来管理命令行参数，更加方便修改参数。

因为PCNN+ONE和PCNN+ATT的训练，测试方法不太一样，因此为了简单起见， 分别写了主文件: `main_mil.py`与`main_att.py`。

训练方式一样，如使用PCNN+ONE 训练大数据集, 后面可以直接修改参数, 默认使用`config.py`的参数:

```

python main_mil.py train --data="NYT"  --batch_size=128

```

注：需要提前按照下一节处理下数据（主要是生成npy格式的数据，方便直接被模型导入).



## 数据预处理

为了节省空间， 上传了LARGE和SMALL两份的原生数据，因此需要用数据预处理下，从而生成npy格式数据。

首先下载两份原始数据，地址:

[百度网盘](https://pan.baidu.com/s/1MyCFWzy89OkBoxfrdHwSfg)  [谷歌云盘](https://drive.google.com/drive/folders/1kqHG0KszGhkyLA4AZSLZ2XZm9sxD8b58?usp=sharing)

数据格式简单说明:
- 第一行: 两个实体ID:  ent1id ent2id
- 第二行: bag标签和bag内句子个数，其中由于少数bag有多个label(不会超过4个)，因此句子label用4个整数表示，-1表示为空，如: 2 4 -1 -1 3 表示该bag的标签为2和4，然后包含3个句子
- 后续几行表示该bag内的句子


将两个zip放到`dataset`目录下，解压，这样会形成两个目录 ，一个NYT, 一个FilterNYT, 其中LARGE数据集在NYT目录，SMALL数据在FilterNYT内，这里的原始数据分别是从Zeng 2015 以及 Lin2016 的开源代码中获得。



对于LARGE数据:



- 切换到NYT目录下，

- 编译执行extract_cpp目录的extract.cpp: `g++  extract.cpp -o extract`, 之后执行:`./extract`, 得到`bag_train.txt, bag_test.txt` (在NYT目录内)，该cpp是Lin2016预处理的代码

- 切换回主目录：执行数据预处理: `python dataset/nyt.py` 这样就会在NYT目录下生成一系列的npy文件。



对于SMALL数据

- 直接执行 `python dataset/filternyt.py` 即可在FilterNYT的目录下生成npy文件。



生成的NPY文件，均使用Pytorch的Dataset来直接导入，具体代码见 `nyt.py` 与`filternyt.py` 的 `*Data`类.

数据预处理完毕之后，即可按照上述的命令来训练/测试。



##  调参优化

在复现的过程了花了不少功夫，踩了不少坑，简单记一下:

- 优化函数使用`Adadelta`而不是`Adam`, 用`SGD` 也可以，不过不如`Adadelta` 效果好。

- Zeng 2015的theano代码中，关于select instance 和predict的地方，有些错误（并没有取概率最大的instance)

- BatchSize相对大一些效果要好（128）



关于结果的说明可以在博客查看。



## 参考

- [PCNN+ONE Zeng 2015](https://github.com/smilelhh/ds_pcnns)
- [PCNN+ATT Lin 2016](https://github.com/thunlp/OpenNRE)
- [RE-DS-Word-Attention-Models](https://github.com/SharmisthaJat/RE-DS-Word-Attention-Models)
- [GloRE](https://github.com/ppuliu/GloRE)
