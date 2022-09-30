---
title: "[boostcamp AI Tech][PyTorch] Lecture 10: PyTorch Troubleshooting"
date: 2022-09-30 13:00:00 + 0900
categories: [boostcamp AI Tech, Week 2 - PyTorch]
tags: [boostcamp, pytorch, level 1, week 2] # TAG names should always be lowercase
math: true
---

- [Going into the Lecture](#going-into-the-lecture)

# Out of Memory(OOM)이 해결하기 어려운 이유들

1. 왜 발생했는지 알기 어렵다
2. 어디서 발생했는지 알기 어렵다
3. Error backtracking이 이상한데로 간다 (GPU는 거짓말을 한다)
4. 메모리의 이전 상황을 파악이 어렵다.

- 1차원적 해결방법: batch size를 줄이고 GPU를 clean (colab에서는 relaunch) 다시 run.

# GPUUtil 사용하기

- nvidia-smi 처럼 GPU의 상태를 보여주는 모듈이다
- Colab은 환경에서 GPU 상태를 보여주기가 편하다
- iteration 마다 메모리가 늘어나는지 확인이 가능하다

```python
  !pip install GPUtil

  import GPUtil
  GPUtil.showUtilization
```

![](/assets/img/boostcamp/2022-09-30-21-11-00.png)

# torch.cuda.empty_cache() 써보기

- 사용되지 않은 GPU상 cache를 정리하여 가용 메모리를 확보한다
- del과는 구분이 필요하다 (del은 관계만 끊는다)
- reset 대신 쓰기 좋은 함수이다.

```python
# del vs torch.cuda.empty_cache()
import torch
from GPUtil import showUtilization as gpu_usage

print("Initial GPU Usage")
gpu_usage()

tesnorList = []
for x in range(10):
    tensorList.append(torch.randn(10000000,10),cuda())

print("GPU Usage after allocating a bunch of Tensors")
gpu_usage()

del tensorList

print("GPU Usage after deleting the Tensors")
gpu_usage()

print("GPU Usage after emptying the cache")
torch.cuda.empty_cache()
gpu_usage()
```

![](/assets/img/boostcamp/2022-09-30-21-18-05.png)

# training loop에 tensor로 축적 되는 변수는 확인할 것

- tensor로 처리된 변수는 GPU 상에 메모리를 사용한다
- 해당 변수가 loop안에 연산이 있을 때 GPU에 computational graph를 생성한다(메모리 잠식).

```python
# don't do
total_loss = 0
for i in range(10000):
    optimizer.zero_grad()
    output = model(input)
    loss = criterion(output)
    loss.backward()
    optimizer.step()
    total_loss += loss # 이럴 경우에는 loss_1 + loss_2 + loss_3... 로 저장이 된다
```

- 1-d tensor의 경우에는 python 기본 객체로 변화하여 처리를  해야한다.

```python
# do
total = loss = 0

for x in range(10):
    #assume loss is computed
    iter_loss = torch.randn(3, 4).mean()
    iter.loss.requires_grad = True
    total_loss += iter_loss.item
```

# del 명령어를 적절히 사용하기

- 필요가 없어진 변수는 적절한 삭제가 필요하다
- python의 메모리 배치 특성상 loop이 끝나도 메모리를 차지한다.

```python
for x in range(10):
    i = x

print(i) # 9 is printed
```
# 가능 batch 사이즈를 실험해보기

- 학습시 OOM이 발생했다면 batch 사이즈를 1로 해서 시험해보기

```python
oom = False
try:
    run_model(batch_size)
except RuntimeError: # Out of Memory
    oom =  True

if oom:
    for _ in range(batch_size):
        run_model(1)
```

# torch.no_grad() 사용하기

- Inference 시점에서는 touch.no_grad() 구문을 사용하기
- bachward pass으로 인해 쌍이는 메모리에서 자유로워진다.

```python
with torch.no_grad()
    for data, target in test_loader:
        output = network(data)
        test_loss += F.nll_loss(....)
        ...
```

# 예상치 못한 에러 메세지

- OOM 말고도 유사한 에러들이 발생한다
- CUDNN_STATUS_NOT_INIT이나 device-side-assert등이 있다
- 해당 에러도 cuda와 관련하여 OOM의 일종으로 생각될 수 있으며, 적절한 코드 처리가 필요하다.

# 그 외

- colab에서 너무 큰 사이즈는 실행하지 말 것
  - (linear, CNN, **LSTM**)
- CNN의 대부분의 에러는 크기가 안 맞아서 생기는 경우 (touchsummary등으로 사이즈를 맞출 것)
- tensor의 float precision을 16bit로 줄일 수도 있다
- 결국엔 구글링이 답이다.

# Further Reading

- [GPU 에러 정리](https://brstar96.github.io/devlog/shoveling/2020-01-03-device_error_summary/)
- [OOM시에 GPU 메모리 flush하기](https://discuss.pytorch.org/t/how-to-clean-gpu-memory-after-a-runtimeerror/28781)
- [PyTorch에서 자주 발생하는 에러 질문들](https://pytorch.org/docs/stable/notes/faq.html)

-----------------------------------


