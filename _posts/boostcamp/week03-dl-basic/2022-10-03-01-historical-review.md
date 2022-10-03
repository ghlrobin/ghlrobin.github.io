---
title: "[boostcamp AI Tech][DL Basic] Lecture 1: Historical Review"
date: 2022-10-03 09:00:00 + 0900
categories: [boostcamp AI Tech, Week 3 - DL Basic]
tags: [boostcamp, dl basic, level 1, week 2] # TAG names should always be lowercase
math: true
---

- [Going into the Lecture](#going-into-the-lecture)

# Going into the Lecture

- Key components of Deep Learning
  1. The **data** the model can learn from
  2. The **model** how to transform the data
  3. The **loss** function that quantifies the badness of the model
  4. The **algorithm** to adjust the parameters to minimize the loss

- Whenever looking at research papers, consider these for components to see more clearly what the paper is contributing to

## Data

- Data depend on the type of the problem to solve.
- Classification: classify the image
- Semantic Segmentation: partition the image to different objects by pixel
- Detection: Find bounding boxes of objects in images
- Pose estimation: Find 3D or 2D skeleton information
- Visual QnA: Answer a question concerning the image in question

## Model

- AlexNet, GoogLeNet, LSTM, Deep AutoEncoders, GAN 등 다양한 모델들이 있다

## Loss

- The loss function is a proxy of what we want to achieve. (Not exactly what we want)
- Regression Task: Mean Square Error(MSE)
- Classification Task: Cross Entropy(CE)
- Probabilitistic Task: Maximum Likelihood Estimator(MLE)

## Optimization Algorithm

- SGD, Momentum, NAG, Adagrad, Adadelta, RMSprop
- Regularizations: Dropout, Early Stopping, K-fold Validation, Weight Decay, Batch Normalization, MixUp, Ensemble, Bayesian Optimization

# Historical Review

- Follows Denny Britz article [(source)](https://dennybritz.com/posts/deep-learning-ideas-that-stood-the-test-of-time/)

## 2012 - AlexNet

![](/assets/img/boostcamp/2022-10-03-09-30-19.png)

- 224 x 224 이미지를 분류하는 대회 ILSVRC에서 최초로 DL 모델로 1등을 했고 DL의 시대가 열리게 되었다.

# 2013 - DQN

![](/assets/img/boostcamp/2022-10-03-09-30-40.png)

- 알파고 vs 이세돌의 시작이 DQN이다
- 딥마인드의 연구결과
- 아탙리라는 블록깨기 게임을 강화학습을 이용해 풀어내려고 만든 모델
- Q-Learning이라는 방슥을 DL과 접목시켜서 학습했다.

# 2014 - Encoder / Decoder

![](/assets/img/boostcamp/2022-10-03-09-32-41.png)

- 단어의 연속이 주어졌을 때 다른 언어의 단어의 연속으로 출력하는데 사용한다
- Sequence-to-Sequence(Seq-2-Seq)를 할 수 있게 되었다.

# 2014 - Adam Optimizer

![](/assets/img/boostcamp/2022-10-03-09-34-15.png)

- 그냥 잘 작동한다

# 2015 - Generative Adversarial Network(GAN)

![](/assets/img/boostcamp/2022-10-03-09-36-55.png)

- Network가 스스로 학습 데이터(generator, discriminator)를 만들어내서 학습을 한다
- 중요하다.

# 2015 - Residual Networks(ResNet)

- 예전에 딥러닝은 layer가 너무 많으면 성능이 떨어졌었다
- Network를 깊게 쌓아도 문제가 발생하지 않게 했다.

# 2017 - Transformer

![](/assets/img/boostcamp/2022-10-03-09-41-02.png)

- Attention Is All You Need이라는 논문으로 발표를 했다
- Transfer가 기존의 RNN의 분야에서 RNN을 다 대체를 하고 vision까지 넘보고 있다
- 매우 중요하다
- 어떤 장점이 있고 왜 좋은 성능을 내는지 알아야한다.

# 2018 - BERT(fine-tuned NLP models)

![](/assets/img/boostcamp/2022-10-03-09-43-11.png)

- Bidirectional Encoder Representations from Transformers
- 학습 데이터(news)가 많지 않을 때 큰 규모의 corpus(wikipedia)를 사용해서 pre-training

# 2019 - Big Language Models: GPT-X

![](/assets/img/boostcamp/2022-10-03-09-48-18.png)

- BERT의 끝판왕
- parameter가 굉장히 많아서 붙어진 이름이다 (billion 단위)


# 2020 - Self Supervised Learning

![](/assets/img/boostcamp/2022-10-03-09-48-50.png)

- unlabeled data를 사용해서 학습을 한다
- supervised랑 unsupervised 사이라고 할 수 있다
- 스스로 lavel을 만든다
- audio processing, speech recognition에서 좋은 성능을 보이고 있다.

-----------------------------------


