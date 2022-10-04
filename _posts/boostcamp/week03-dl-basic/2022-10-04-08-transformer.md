---
title: "[boostcamp AI Tech][DL Basic] Lecture 8: Transformer"
date: 2022-10-04 11:00:00 + 0900
categories: [boostcamp AI Tech, Week 3 - DL Basic]
tags: [boostcamp, dl basic, level 1, week 3] # TAG names should always be lowercase
math: true
---

- [Sequential Model](#sequential-model)

# Transformer

![](/assets/img/boostcamp/2022-10-04-10-51-46.png)

- Transformer is the first sequence transduction model based entirely on attention
- Proccesses sequential data and encodes. Not only works for NLP but for visual transformer, text-to-image (DALL-E)

Things we need to understand:

1. Given n words how are they processed into the encoder?
2. What is the information being sent from encoders to decoders?
3. How does decoders generate the output?

# 1. Given n words how are they processed into the encoder?

![](/assets/img/boostcamp/2022-10-04-10-57-46.png)

- In an encoder there are two processes 1) self-attention and 2) feed forward neural network
- The **Self-Attention** in both encoder and decoder is the cornerstone of Transformer
  - Transformer encodes each word to feature vectors with Self-Attention (where there are dependencies between the input words)
- Self-Attention at high level
  - The animal didn't cross the street because it was too tired
  - Self-Attention figures out what "it" is

![](/assets/img/boostcamp/2022-10-04-11-51-14.png)

- Self-Attention creates three vectors (3 NN) Queries, Keys, Values per each word
- Embedding vector $x_1$ is converted to these three vecotrs
- Then we compute the **score** per each word by inner producting Queries and respective Keys vectors
- Then we compute the attention weights by scaling followed by softmax
- Then the final encoding is done by the weighted sum of the Value vectors.

![](/assets/img/boostcamp/2022-10-04-12-03-27.png)

- In python it's just a one liner
- if there are $n$ words, $n^2$ operations need to be computed at once (Multi-headed attention). This is the downside of Transformers
- To match the output, we simply pass the attention heads(encoded vectors) through additional (learnable) linear map
- In summary this is what we do:

![](/assets/img/boostcamp/2022-10-04-12-09-30.png)

- Why do we need positional encoding?
- To add the postional information of the inputs
- They are added to the origional embedding.

![](/assets/img/boostcamp/2022-10-04-12-12-18.png)

# 2. What is the information being sent from encoders to decoders?

![](/assets/img/boostcamp/transformer_decoding_2.gif)

- Transformer transfers key (K) and value (V) of the topmost encoder to the decoder

- The output sequence is generated in an autoregressive manner.
- The "Encoder-Decoder Attention" layer works just like multi-headed self-attention, except it creates its Queries matrix from the layer below it and takes the Key and Values from the encoder stack.

# Vision Transformer

![](/assets/img/boostcamp/2022-10-04-12-20-46.png)

- Transformer Encoder is used.

# DALL-E

![](/assets/img/boostcamp/2022-10-04-12-21-59.png)

- Transformer Decoder is used.


















-------------------------------