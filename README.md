## opinion_mining
之江杯2019-电商评论观点挖掘

## 整体思路
使用两阶段的pipeline的方式，第一阶段用BIES标注`OpinionTerms`和`Polarities`，第二阶段携带第一阶段抽取的**一个**`OpinionTerms`信息去标注`AspectTerms`(如果这个`OpinionTerms`没有对应`AspectTerms`，将`AspectTerms`的序列标注置为全O)，同时使用一个分类器去得到这个`OpinionTerms` `AspectTerms` pair或者`OpinionTerms`的`Categories`，两阶段的训练都是采用`multi-task`。为什么第二阶段不用BIES同时标注类别?因为没有`AspectTerms`的情况很多，但是又必须输出一个`Categories`。因为做这个比赛的时间比较赶，没有对两个阶段的总体做线下评分，都是看两阶段有提升就提交了，最后复赛排名30。

## 领域迁移
一阶段因为只抽取`OpinionTerms`和`Polarities`，直观上来看，这个任务是不用区分领域的(只是抽取makeup与laptop的`OpinionTerms`和`Polarities`)，所以只是简单的混合了两个领域的的数据(laptop makeup)，第二阶段laptop和makeup的`Categories`，就有一定的差距了，目前我的解决方案是先使用大数据量的makeup数据去训练第二阶段，然后加载权重，更换最后的`Categories`分类器，再小学习率的微调laptop的数据。

## 联合训练的模型
之前受到过苏神信息抽取任务的启发，曾经尝试过将两个阶段合并到一起。每次训练的时候，模型1还是像上述一阶段的任务这样训练，模型2每次采样一个`OpinionTerms`去抽取`AspectTerms`。（在每一个epoch都重新采样，所以epoch数目足够的话，还是可以见到很多`AspectTerms`数据的），但是我试了一下效果并不好，线下的指标很低，猜测是简单的随机采样可能采到负样本的情况比较多，后面应该尝试更多的采样策略。

## Keras显示多个loss
因为使用了多任务训练，用hook的方式，让Keras能在训练过程中显示每一个任务的loss，详情见代码。
`tensorflow-gpu==1.8.0 keras==2.2.4`

## reference
- [多任务训练及数据处理参考](https://github.com/EliasCai/viewpoint-mining)
- [Keras中无损实现复杂（多入参）的损失函数](https://zhuanlan.zhihu.com/p/54024591)
