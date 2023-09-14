---
layout: post
title: 'Semantic Match Deep Dive 1'
date: 2023-09-14
excerpt: "A detailed introduction of semantic match on both sentence to sentence(s2s) and sentence to passage(s2p) tasks."
tags:
  - NLP
---

# 语义匹配
语义匹配是NLP领域的基础任务之一，其任务目标是判断两个句子是否具有相同的语义。语义匹配在自然语言处理中有着广泛的应用，例如问答系统、信息检索、机器阅读理解等。

语义匹配是NLP领域的基础任务之一，其任务目标是判断两个句子是否具有相同的语义。语义匹配在自然语言处理中有着广泛的应用，例如问答系统、信息检索、机器阅读理解等。

语义匹配任务可以分为两个子任务：

- 句子到句子语义匹配（S2S）
- 句子到段落语义匹配（S2P）

## 句子到句子语义匹配（S2S）
句子到句子语义匹配（S2S）任务是判断两个句子是否具有相同的语义。S2S任务通常用于问答系统的意图召回，例如电商，证券，银行，医疗中的常见FAQ的语义匹配。这一类都是短文本的语义匹配，例如“买房子多少钱”和“房子要多少钱”的语义匹配，一般语义之间表达的意思都接近，且语义短，因此这类语义匹配任务相对简单。

## 句子到段落语义匹配（S2P）
句子到段落语义匹配（S2P）任务是判断句子到段落之间是否存在语义关系。由于句子的长短存在偏差，所以在进行向量投影时，短句子会存在一定程度的偏差。因此，S2P任务在语义匹配中具有更高的难度。

## 语义匹配模型
常见的语义匹配模型有两种，一种是基于表征式（Representation-based）的语义匹配模型，另一种是基于交互式（Interaction-based）的语义匹配模型。

### 基于表征式的语义匹配模型 
<img src='/images/semantic_match/sentence-bert.jpg'>
基于表征式的语义匹配模型的代表是 Sentence—Bert[^1] 的双塔模型，左边是训练过程，对句子进行进行一层Bert输出，经过Pooling得到对应A、B句的向量u、v，再通过concatenate u,v, |u-v| 再乘以一个可训练的权重经过softmax的到最后的结果(公式如下)。右边是推理的过程，用模型进行推理得到A、B向量，u、v再进行余弦相似度计算，通过设定一定的阈值求出最优的结果。

$$ o=softmax(W_t(u,,v,|u-v|)), W_t\in R^{3n\times k} $$

Sentence Bert 这种框架有两种实现，Bi-encoder 和 Dual-encoder。
- Bi-encoder首先将Query和Candidate输入到共享的Transformer编码器中，将其分别编码成向量表示，然后通过计算两个向量之间的相似度得分，来判断Query和Candidate的匹配程度。典型的有近期Moka AI发布的M3E模型，在s2s(sentence to sentence)和s2p(sentence to passage)任务上表现不俗，超过了之前比较优秀的[text2vec](https://huggingface.co/shibing624/text2vec-base-chinese)模型 (large的模型可以看这里)，具体可以看一下[hugging face](https://huggingface.co/moka-ai/m3e-base)的介绍。


- Dual-encoder模型的基本架构是将Query和Candidate分别输入不同的Transformer编码器中，分别编码成向量表示，然后计算Query和Candidate的相似度得分，来判断Query和Candidate的匹配程度。这种适用于两种句子维度不一样的匹配任务，类似于query匹配文档，query匹配答案等。

在Bi-encoder模型中，Query和Candidate共享了同一个编码器，因此它们在表示空间中具有相同的维度和分布。而Dual-encoder模型则使用了两个独立的编码器，因此它们在表示空间中具有不同的维度和分布。这种差异可能对模型的表现产生一定的影响。同时，由于Dual-encoder模型中的两个编码器具有独立的参数空间和状态空间，因此Dual-encoder模型可以对Query和Candidate的特征进行更灵活的处理和提取。然而，由于需要用到两个编码器，Dual-encoder模型的训练和推理成本通常比Bi-encoder模型高。

### 基于交互式的语义匹配模型
<img src='/images/semantic_match/cross-encoder.jpg'>
交互式匹配方案如上右图，将两段文本拼接在一起当成单个文本进行分类，交互式由于使得两个文本能够进行充分的比较，所以它准确性通常较好，但在检索场景中使用效率低下，因为需要现场推理向量，而特征式可以提前将所有的Candidates进行计算并缓存，在检索过程中只需要对Query进行向量推理，然后匹配所有的Candidates就行了，但相对而言，特征式交互程度较浅，效果通常比交互式要差。

### 多阶段召回
更为常见的做法为利用表征式语义匹配模型的速度优势召回top-n候选句子，然后将这些句子和Query进行匹配，再进行排序，最后召回top-k的句子。

- 召回阶段：Query在线通过语义模型计算句子向量，然后和向量库中的句子计算余弦相似度，选出打分为top-n的句子作为候选集
- 排序阶段：将Query和候选集中的句子进行拼接，然后传入Cross Encoder，计算分数，排序。

## 论文推荐
1. Dense Passage Retrieval for Open-Domain Question Answering[^2]
2. RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering[^3]
2. Unsupervised Corpus Aware Language Model Pre-training for Dense Passage Retrieval[^4]
3. HLATR: Enhance Multi-stage Text Retrieval with Hybrid List Aware Transformer Reranking[^5]

### Dense Passage Retrieval
<img src="/images/semantic_match/dpr.jpg">
Dense Passage Retrieval算法，这个算法使用的dual-encoder分别对query和passages进行编码，然后计算相似度从而更新网络。这里用到的loss也相对比较简单，就是negative log likelihood。

$$L(q_i, p_i^{+},p_{i,1}^{-},p_{i,2}^{-},...,p_{i,n}^{-}) = -log\frac{e^{sim(q_i, p_i^{+})}}{sim(q_i, p_i^{+}) + \sum_{j=1}^{n}e^{sim(q_i, p_{i,j}^{-})}}  $$

主要一些比较新颖的方式是对于negatives的sampling。一共三种方式也比较好理解：

- 随机取
- BM25算法取topk，然后再从中去掉原本query的答案。
- in-batch negatives：简单来说如果batch_size=64，那么就能有63个negatvies，即除了query本身的positive，其他都可以作为negatives。

### RocketQA
Dense passage retrieval下的训练和推理之间的差异，即训练过程中只是选取了部分的样本作为负例，而推理过程中则是对所有的样本进行比较。同时，在进行训练的过程中negative samples往往存在了大量的false negative samples，即标注为非答案文本的文段也是可以作为答案的。

针对上面这两个问题，文章提出了三个优化策略Cross-batch negatives, Denoised Hard Negatives, Data Augmentation.

Cross-batch negatives主要做的就是将m台上的GPU机器上的n个samples都进行embeddings计算，然后下发到每一个GPU机器上，那么每一个训练样本都有m*n-1个负样本，对比之前的in-batch negatives的n-1个负样本，这样极大地增加了负例样本的学习。

Denoised Hard Negatives，先train一个dual encoder来召回negatives， 然后再训练一个cross encoder来去除false negatives，这样让模型学习到负样本尽可能是对的，相当于是做了一次数据清洗。

Data Augmentation就是用已经训练好的cross encoder对一些未标注数据进行标注，类似于semi-supervised learning，来增大数据量。

这种方式在测评数据集上的表现不俗，但是对于机器的要求比较高，就比如第一步的Cross-batch negatives来说，这需要尽可能地增加机器的数目，同时对于单个GPU机器来说，增加了大批量的negatives，对于GPU的显存来说是一个很大的挑战。后面的train dual encoder的方式也是一个多阶段的训练，相对来说训练的成本比较高。

### coCondense
Condenser 预训练架构，它通过 LM 预训练学习将信息压缩到密集向量中。最重要的是，作者进一步提出了 coCondenser，它添加了一个无监督的语料库级对比损失来预热段落嵌入空间。它显示出与 RocketQA 相当的性能，RocketQA 是最先进的、经过精心设计的系统。coCondense使用简单的小批量微调，无监督学习，即随机从一笔文档中抽出文本片段，然后训练模型。目标是让相同文档出来的CLS的embedding要尽可能的相似，而来自不同的文档出来embedding的要尽可能不相近。

### HLATR
<img src="/images/semantic_match/hlatr.jpg">
先检索再排序是一种比较常用的文本检索手段，但是常见的做法通常是只关注于优化某一个阶段的模型来提升整体的检索效果。但是直接将多个阶段耦合起来进行优化的却还没有被很深入的研究。作者提出了一个轻量级的HLATR框架，可以高效地进行检索，并在两个大数据集上进行了验证。这里作者提到两个模型虽然都是进行排序，但是模型的关注的点不一样，表征式的模型（retriever）偏向于粗颗粒度特征，交互式的模型（interaction）偏向于query和document之间的信息交互。同时作者做了一个简单的weighted combination，就是给予召回和排序这两个阶段不同的权重，对于整体的召回效果也是有提升的。

## References
[^1]: [Sentence-Bert:Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/pdf/1908.10084.pdf)    
[^2]: [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906)
[^3]: [RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering](https://aclanthology.org/2021.naacl-main.466/)
[^4]: [Unsupervised Corpus Aware Language Model Pre-training for Dense Passage Retrieval](https://arxiv.org/pdf/2108.05540.pdf)
[^5]: [HLATR: Enhance Multi-stage Text Retrieval with Hybrid List Aware Transformer Reranking](https://arxiv.org/pdf/2205.10569.pdf)

