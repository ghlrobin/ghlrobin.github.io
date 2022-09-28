---
title: "[boostcamp AI Tech][PyTorch] Lecture 5: PyTorch Dataset"
date: 2022-09-28 11:00:00 + 0900
categories: [boostcamp AI Tech, Week 2 - PyTorch]
tags: [boostcamp, pytorch, level 1, week 2] # TAG names should always be lowercase
math: true
---

- [Going into the Lecture](#going-into-the-lecture)
- [모델에 데이터를 먹이는 방법](#모델에-데이터를-먹이는-방법)
- [Dataset 클래스](#dataset-클래스)
- [Dataset 클래스 생성시 유의점](#dataset-클래스-생성시-유의점)
- [DataLoader 클래스](#dataloader-클래스)
- [Casestudy](#casestudy)

# Going into the Lecture

- 요즘에 대용량 데이터를 어떻게 잘 넣어주는 것이 더 중요해졌다
- 이걸 해주는 것이 PyTorch Dataset API
- 이번 강의에 어떻게 기존 파일 형태에서 -> 데이터 feeding 해주는지 설명한다.

# 모델에 데이터를 먹이는 방법

![](/assets/img/boostcamp/2022-09-28-15-01-24.png)

- Dataset Class의 ```getitem``` 하나의 데이터를 가져올 때 어떻게 데이터를 반활할지 선언해준다
- transforms에서 tensor로 데이터 변환
- dataloader는 data를 묶어서 (섞어주거나) 모델에 feeding.

# Dataset 클래스

- 데이터 입력 형태를 정의하는 클래스
- 데이터를 입력하는 방식의 표준화
- Image, Text, Audio 등에 따른 다른 입력정의.

```python
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
	def __init__(self, text, labels):
    self.labels = labels
		self.data = text
	
	def __len__(self):
		return len(self.labels)
	
	def __getitem__(self, idx):
		label = self.labels[idx]
		text = self.data[idx]
		sample = {"Text": text, "Class": label}
		return sample
```

# Dataset 클래스 생성시 유의점

- data 형태에 따라 각 함수를 다르게 정의함
- 모든 것을 데이터 생성 시점에 처리할 필요는 없음
  - image의 tensor 변화는 학습에 필요한 시점에 변활
- 데이터 셋에 대한 표준화된 처리방법 제공 필요
  - 후속 연구자 또는 동료에게는 빛과 같은 존재
- 최근에 HuggingFace등 표준화된 라이브러리 사용.

# DataLoader 클래스

- Data의 Batch를 생성해주는 클래스 (묶어준다)
- 학습직전(GPU feed전) 데이터의 변환을 책임
- Tensor로 변환 + Batch 처리가 메인 업무
- 병렬적인 데이터 전처리 코드의 고민 필요
- [link](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) to pytorch doc.

```python
DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0, workers_init_fn=None, *, prefetch_factor=2 persistent_workers=False)
```
# Casestudy

- 데이터 다운로드부터 loader까지 직접 구현해보기
- NotMNIST 데이터의 다운로드 자동화 도전.

- - - 


