---
title: "[boostcamp AI Tech][PyTorch] Lecture 9: Hyperparameter Tuning"
date: 2022-09-30 12:00:00 + 0900
categories: [boostcamp AI Tech, Week 2 - PyTorch]
tags: [boostcamp, pytorch, level 1, week 2] # TAG names should always be lowercase
math: true
---

- [Going into the Lecture](#going-into-the-lecture)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Grid vs Random Layout](#grid-vs-random-layout)
- [Ray](#ray)

# Going into the Lecture

- 모델 성능을 좋게 만드는 법은 크게 3가지가 있다
  1. 모델 바꾸기: 이미 고정된 좋은 모델을 사용한다 (RESNET, CNN, Transformer)
  2. 데이터 바꾸기: 가장 좋은 방법이다
  3. hyperparameter tuning.

# Hyperparameter Tuning

- 모델 스스로 학습하지 않는 값을 우리가 지정해야 한다. (Learning rate, 모델의 크기, optimizer 등등)
- 한때는 하이퍼 파라메터에 의해서 값이 크게 좌우 될 때가 있었지만 요즘은 극도로 많은 데이터로 완화를 했다
- 당연히 하지만 중요도가 좀 낮아졌다. (옛날에는 '손맛'이라고...)

# Grid vs Random Layout

- 기본적으로 grid, [random search](https://dl.acm.org/doi/pdf/10.5555/2188385.2188395)가 있는데 최근에는 [베이지안 기반](http://proceedings.mlr.press/v80/falkner18a/falkner18a.pdf) 기법들을 많이 사용한다

![](/assets/img/boostcamp/2022-09-30-20-47-43.png)

- grid layout 때는 log를 취해서 값을 올려준다.

# Ray

- multi-node multi processing 지원 모듈
- ML/DL의 병렬 처리를 위해 개발된 모듈
- 기본적으롤 현재의 분산병렬 ML/DL 모듈의 표준
- Hyperparameter Search를 위한 다양한 모듈을 제공한다.

```python
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

data_dir = os.path.abspath("./data")
load_data(data_dir)
config = {
    "l1": tune.sample_from(lambda: 2**np.random.randint(2,9)),
    "l2": tune.sample_from(lambda: 2**np.random.randint(2,9)),
    "lr":tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([2,4,8,16]) # config search  space 지정
}

scheduler = ASHASceduler(metric=metric, mode=mode, max_t=max_num_epochs, grace_period=1, reduction_factor=2) # 학습 스케줄링 알고리즘 지정
reporter = CLIReporter(metric_colums=["loss", "accuracy", "training_iteration"]) # 결과 출력 양식

result = tune.run(
    partial(train_cifar, data_dir=data_dir),
    resource_per_trial={'cpu':2, 'gpu':gpus_per_tiral},
    config=config, num_samples=num_samples,
    scheduler=scheduler,
    progress_reporter=reporter) # 병렬 처리 양식으로 학습 시행
```
-----------------------------------


