---
title: "[boostcamp AI Tech][PyTorch] Lecture 4: AutoGrad & Optimizer"
date: 2022-09-28 10:00:00 + 0900
categories: [boostcamp AI Tech, Week 2 - PyTorch]
tags: [boostcamp, pytorch, level 1, week 2]	# TAG names should always be lowercase
math: true
---

- [torch.nn.Module](#torchnnmodule)
- [nn.Parameter (class)](#nnparameter-class)
- [Backward](#backward)
- [optimizer.step](#optimizerstep)
- [optimizer.zero_grad()](#optimizerzero_grad)


# torch.nn.Module

- 딥러닝을 구성하는 Layer의 base class
- input, output, Forward 정의
  - Backward는 AutoGrad으로 자동화되어 있음
- 학습의 대상이 되는 parameter(tensor) 정의

- - -
# nn.Parameter (class)

- Tensor 객체의 상속 객체
- nn.Module 내에 attribute가 될 때는 require_grad=True로 지정되어 학습 대상이 되는 Tensor
- 우리가 직접 지정할 일은 잘 없음
  - 대부분의 layer에는 weights 값들이 지정되어 있음

# Backward

- Layer에 있는 Parameter들의 미분을 수행
- Forward의 결과값 (model의 output=예측치)과 실제값간의 차이(loss)에 대해 미분을 수행
- 해당 값으로 Parameter 업데이트

# optimizer.step

- backward를 통해 업데이트된 parameter값으로 gradient descent 진행

# optimizer.zero_grad()

- clear gradient buffers because we don't want any gradient from previous epoch to carry forward