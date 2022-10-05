---
title: "[boostcamp AI Tech][Data Viz] Lecture 01: Introduction to Visualization"
date: 2022-10-05 10:00:00 + 0900
categories: [boostcamp AI Tech, Week 3 - Data Visualization]
tags: [boostcamp, dl basic, level 1, week 3] # TAG names should always be lowercase
math: true
---

- [데이터 시각화란 무엇일까?](#데이터-시각화란-무엇일까)

# 데이터 시각화란 무엇일까?

다양한 요소가 포함된 작업이다
1. 목적: 왜 시각화를 하나요?
2. 독자: 시각화 결과는 누구를 대상으로 하나요?
3. 데이터: 어떤 데이터를 시각화할 것인가요?
4. 스토리: 어떤 흐름으로 인사이트를 전달할 것인가요?
5. 방법: 전달하고자 하는 내용에 맞게 효과적인 방법을 사용하고 있나요?
6. 디자인: UI에서 만족스러운 디자인을 가지고 있나요?

# 데이터 이해하기

- 데이터 시각화를 위해서는 데이터가 우선적으로 필요하고 시각화를 진행할 데이터를 데이터셋 관점(global)에서 또는 개별 데이터의 관점(local)에서 정리할지를 정해야 한다.

## 데이터셋의 종류

- 요즘에는 수많은 데이터셋이 존재를 한다.
  - 정형 데이터
  - 시계열 데이터
  - 지리 데이터
  - 관계형(네티워크) 데이터
  - 계층적 데이터
  - 다양한 비정형 데이터

- 대표적으로 데이터의 종류는 4가지로 분류한다
- 수치형(numerical)
  - 연속형(continuous): 길이, 무게, 온도 등
  - 이산형(discrete): 주사위 눈금, 사람 수 등
- 범주형(categorical)
  - 명목형(nominal): 형액형, 종교 등
  - 순서형(ordinal): 학년, 벌점, 등급 등

## 정형 데이터

- 테이블 형태로 제공되는 데이터이며 일반적으로 csv, tsv 파일로 제공이 된다
- Row가 데이터 1개 item
- Column은 attribute(feature)
- 가장 쉽게 시각화할 수 있는 데이터셋이다
  - 통계적 특성과 feature 사이 관계
  - 데이터 간 관계
  - 데이터 간 비교
- 가장 많이 다루게 될 데이터셋이다.

## 시계열 데이터 (Timeseries)

- 시간 흐름에 따른 데이터를 time-series 데이터라고 한다
- 기온, 주가 등 정형데이터와 음성, 비디오와 같은 비정형 데이터가 존재 한다
- 시간 흐름에 따른 추세(Trend), 계절성(Seasonality), 주기성(Cycle)등을 살필 수 있다.

## 지리 데이터

- 지도 정보와 보고자 하는 정보 간의 조화 중요와 지도 정보를 단순화 시키는 경우도 존재한다
- 거리, 경로, 분포 등 다양한 실사용이 가능하다
- 실제로 어떻게 사용하는지가 중요하다.

## 관계 데이터

- 객체와 객체 간의 관계를 시각화한다
  - Graph Visualization / Network Visualization
- 객체는 Node로, 관계는 link로 표현을 한다
- 크기, 색, 수 등으로 객체와 관계의 가중치를 표현을 한다
- 휴리스틱하게 노드 배치를 구성한다

## 계층적 데이터

- 관게 중에서도 포함관계가 분명한 데이터이다
  - 네트워크 시각화로도 표현 가능
- Tree, Treemap, Sunburst 등으로 표현이 가능하다

# 시각화 이해하기

- A **mark** is a basic graphical element in an image (points, lines, areas)
- A visual **channel** is a way to control the apperance of marks, independent of the dimensionality of the geometric primitive
  - position(horizontal, vertical or both), color, shape, tilt, and size all have an impact on visualization

## 전주의적 속성(pre-attentive attribute)

![](/assets/img/boostcamp/2022-10-05-10-32-41.png)

- 주의를 주지 않아도 인지하게 되는 요소들을 말한다
  - 시각적으로 다양한 전주의적 속성이 존재한다

- 하지만 동시에 사용하면 인지하기가 어렵다
  - 적절하게 사용할 때, 시각적 분리(visual pop-out)















-------------------------------