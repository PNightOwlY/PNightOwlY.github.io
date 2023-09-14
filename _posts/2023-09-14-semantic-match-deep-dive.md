---
layout: post
title: 'Semantic Match Deep Dive 1'
date: 2023-09-14
excerpt: "A detailed introduction of semantic match on both sentence to sentence(s2s) and sentence to passage(s2p) tasks."
tags:
  - NLP
---

# Introduction of Semantic Match
Semantic match is one of the basic tasks of NLP, the goal of the task is to determine whether two sentences have the same semantic meaning. Semantic match is widely used in natural language processing, such as question answering, information retrieval, machine reading comprehension, etc.

Semantic match tasks can be divided into two categories:
- sentence to sentence（S2S）
- sentence to passage（S2P）

## sentence to sentence（S2S）
The goal of sentence to sentence is to determine whether two sentences have the same semantic meaning. S2S tasks are widely used in intent recogniton in question answering system, such as Ecommerce, Finance, Medical etc. This task is semantic matching of short texts, such as "How much does it cost to buy a house?" and "How much does it cost to buy a house?", so this type is relatively easier.


## sentence to passage（S2P）
The sentence to passage is to determine whether there is a semantic relationship between two sentences and the passage. Due to the bias in the length of sentences, there will be a certain degree of bias in short sentences in projection space. 


## Semantic Match Models
There are two common semantic matching models, one is a representation-based semantic matching model, and the other is an interaction-based semantic matching model.

### Represtation-based model
<img src='/images/semantic_match/sentence-bert.jpg'>
The representation-based semantic matching model is represented by the two-tower model of Sentence-Bert[^1]. The left side is training process, the sentence first output by Bert, and then pooling to get the vector u, v, and then concatenate u, v, |u-v|, |u-v| is then multiply by a trainable weight to get final result through softmax. The right side is the inference process, use the model to get the vector of A(u) and B(v), and then get the cosine similarity and filtered by certain threshold.

$$ o=softmax(W_t(u,,v,|u-v|)), W_t\in R^{3n\times k} $$

There are two implementaion of Sentence Bert, Bi-encoder and Dual-encoder respectively.
- Bi-encoder: Compute the query and candidate vector representations with shared transformer encoder, and then compute the cosine similarity between the query and candidate vectors, to determine the similarity of the query and candidate. The typical Bert-like model is M3E, text2vec, BGE.
- Dual-encoder: Compute the query and candidate vector representations with different transformer encoder.

The two encoders in the Dual-encoder model have independent parameter spaces and state spaces, the Dual-encoder model can process and extract the features of Query and Candidate more flexibly. The training and inference costs of Dual-encoder models are usually higher than Bi-encoder models.

### Interaction-based model
<img src='/images/semantic_match/cross-encoder.jpg'>
The interactive matching scheme is as shown on the right, which splices two pieces of text together as a single text for classification. Interactive matching allows two texts to be fully compared, so it performs much better, but it is inefficient in retrieval scenarios due to the on-site inference of vectors is required, and representation-based method can calculate and cache all Candidates in advance. During the retrieval process, only vector  for Query is computed, and then all Candidates are calculate the similarity. However, relatively speaking, the degree of interaction of characteristic formulas is Shallow and generally less effective than interactive.

### Multi-stage Retrieval
The more common way is to use the representation-based method to retrieve top-n sentences, and then use interaction-base method match the Query and top-n sentences to get the final ranking results.

- retrieval stage: calcuate the cos similarity between Query and all sentence, and pick the top-n sentences as candidates.
- ranking stage: concat the Query and top-n sentences as a single text respectively, and use Cross Encoder to get the score of the text.

## Recommend thesis
1. Dense Passage Retrieval for Open-Domain Question Answering[^2]
2. RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering[^3]
2. Unsupervised Corpus Aware Language Model Pre-training for Dense Passage Retrieval[^4]
3. HLATR: Enhance Multi-stage Text Retrieval with Hybrid List Aware Transformer Reranking[^5]

### Dense Passage Retrieval
<img src="/images/semantic_match/dpr.jpg">
Dense Passage Retrieval，use dual-encoder encode query and passages respectively，and compute similarity then update the model's weight. The loss is negative log likelihood as following.

$$L(q_i, p_i^{+},p_{i,1}^{-},p_{i,2}^{-},...,p_{i,n}^{-}) = -log\frac{e^{sim(q_i, p_i^{+})}}{sim(q_i, p_i^{+}) + \sum_{j=1}^{n}e^{sim(q_i, p_{i,j}^{-})}}  $$

The proposed three way to generate negatives:

- random pick
- BM25 to pick top-k，remove the origin answer
- in-batch negatives, batch_size=64，then there is 63 negatvies，except the query's positive, all can be negative.

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

