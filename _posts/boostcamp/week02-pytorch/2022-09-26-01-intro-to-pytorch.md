---
title: "[boostcamp AI Tech][PyTorch] Lecture 1: Intro. To PyTorch"
date: 2022-09-26 8:00:00 + 0900
categories: [boostcamp AI Tech, Week 2 - PyTorch]
tags: [boostcamp, pytorch, level 1, week 2]	# TAG names should always be lowercase
math: true
---

- [이번 주 강의 소개](#이번-주-강의-소개)
- [개요](#개요)
- [Computational Graph](#computational-graph)
- [PyTorch vs Tensorflow](#pytorch-vs-tensorflow)
- [Why PyTorch?](#why-pytorch)
- [Pytorch 특징](#pytorch-특징)


# 이번 주 강의 소개

* 딥러닝 이론을 실제로 구현하기 위해 필요한 PyTorch 프레임워크 사용법에 대해 학습한다
* 기본적인 네트워크 구현 및 데이터 로딩
* 프로젝트 구조
* 로깅, Multi GPU, 이어 학습하기 등의 테크닉 부분들

# 개요

* PyTroch는 딥러닝 framework
* 딥러닝을 바닥부터 짠다? -> 죽을 수도 있다...
* 뷰노 의료 AI 스타트업, 한동대 등 프레임워크를 직접 개발한 적은 있지만 메인 프레임워크로 올라가기에는 한계가 있다
* 만약에 프레임워크가 어떻게 개발되는지 관심이 있다면 밑바닥부터 시작하는 딥러닝 3을 추천한다
* 지금은 남이 만든 걸 사용한다
  * 자료도 많고
  * 관리도 잘되고
  * 표준이라서...
* leading framework으로 facebook의 PyTorch하고 Google의 TensorFlow가 있다
* 두 프레임워크는 computational graph에 차이가 있다

|                          | Keras                             | TensorFlow                     | PyTorch                               |
|--------------------------|-----------------------------------|--------------------------------|---------------------------------------|
| Level of API             | high-level API                    | both high and low level APIs   | Lower-level API                       |
| Speed                    | Slow                              | High                           | High                                  |
| Architecture             | Simple, more readable and concise | Not very easy to use           | Complex                               |
| Debugging                | No need to debug                  | Difficult to debug             | Good debugging capabilities           |
| Dataset Compatibility    | Slow & Small                      | Fast speed & large             | Fast speed & large datasets           |
| Uniqueness               | Multiple back-end support         | Object Detection Functionality | Flexibility & Short Training Duration |
| Created by               | Not a library on its own          | Google                         | Facebook                              |
| Ease of Use              | User-friendly                     | Incomprehensive API            | Integrated with Python                |
| Compuational graphs used | Static Graphs                     | Statich Graphs                 | Dynamic Computation Graphs            |

- - -
# Computational Graph

* 연산의 과정을 그래프로 표현한다
* Define and Run (Static): 그래프를 먼저 정의하고 실행시점에 데이터를 feed한다
  * Build graph once, then run many times
* Define by Run (Dynamic Computational Graph, DCG): 실행을 하면서 그래프를 생성하는 방식
  * Each forward pass defines a new graph (easy to debug)
  * DCG가 느릴 것 같지만 사실 그렇지 않고 "느낌상" DCG가 더 빠르고 편한게 있다.


- - -
# PyTorch vs Tensorflow

* Tensorflow는 구글의 도음을 받아 production, cloud multi-GPU 등의 장점이 있다
* Pytorch는 debugging이 더 쉬워서 research에서 더 많이 활용이 된다

- - -
# Why PyTorch?

* Define by Run의 장점
* 즉시 확인 가능, pythonic code
* GPU support, Good API and community
* 사용하기 편한 장점이 가장 크다

- - -
# Pytorch 특징

* Numpy 구조를 가지는 Tensor 객체로 array 표현한다
* 자동미분(autograd)을 지원하여 DL 연산을 지원한다
* 다양한 형태의 DL을 지원하는 함수와 모델을 지원한다.

- - -