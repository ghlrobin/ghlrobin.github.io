---
title: "[boostcamp AI Tech][PyTorch] Lecture 3: PyTorch 프로젝트 구조 이해하기"
date: 2022-09-26 12:00:00 + 0900
categories: [boostcamp AI Tech, Week 2 - PyTorch]
tags: [boostcamp, pytorch, level 1, week 2]	# TAG names should always be lowercase
math: true
---

# 개요

# Going into the Lecture

- 파이토치의 모듈들이 어떻게 구성되는지
- 프로젝트 코드들이 어떻게 작동하는지
- modules.py, data.py, preprocessor.py 등등

ml 코드는 언제나 Jupyter에서?
- 영원히 세발 자전거를 탈 수 없다
- 개발 초기 단계에서는 대화식 개발 과정이 유리해서 괜찮다
- 하지만 배포 및 공유 단계에서는 notebook 공유의 어려움, 실행순서 꼬임과 재현의 제한이 있다

다른 사람들이 만들어 놓은 프로젝트 template가 많이 존재한다.
- 실행, 데이터, 모델, 설정, 로깅, 지표, 유틸리티 등 다양한 모듈들을 분리하여 프로젝트를 템플릿화
- 우리가 사용할 건 (여기)[http://github.com/victoresque/pytorch-template]

- - - 
