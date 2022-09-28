---
title: "[boostcamp AI Tech][PyTorch] Lecture 2: PyTorch Basics"
date: 2022-09-26 10:00:00 + 0900
categories: [boostcamp AI Tech, Week 2 - PyTorch]
tags: [boostcamp, pytorch, level 1, week 2]	# TAG names should always be lowercase
math: true
---

# 개요

* 기초문법이 numpy랑 비슷하다 (numpy + AutoGrad)
* numpy만 잘 알아도 쉽게 사용이 가능하다 (TensorFlow도 마찬가지)
* autograd 표현이 조금 다르다
* 수식들을 어떻게 쓸 수 있는지 소개를 한다.

- - -
# PyTorch Operations

## Tensor

* Tensor: 다차원 Arrays를 표현하는 PyTorch Class이다
* 사실상 numpy의 ndarray와 동일하다 (TensorFlow의 Tensor와도 동일)
* Tensor를 생성하는 함수도 numpy와 비슷하다


## Numpy to Tensor

numpy:

```python
import numpy as np
n_array = np.arange(10).reshape(2,5)
print(n_array)
print("ndim :", n_array.ndim, "shape :", n_array.shape)
>>> [[0 1 2 3 4]
     [5 6 7 8 9]]
    ndim : 2 shape : (2, 5)
```

pytorch:

```python
import torch
t_array = torch.FloatTensor(n_array)
print(t_array)
print("ndidm :", t_array.ndim, "shape :", t_array.shape)
>>> tensor([[0., 1., 2., 3., 4.],
           [5., 6., 7., 8., 9.]])
    ndim : 2 shape : torch.Size([2, 5])
```

## Array to Tensor

* Tenor 생성은 list나 ndarray를 사용할 수 있다
* `.from_numpy()`, `.tensor()` 또는 `.FloatTensor`로 생성한다
* 하지만 주로 다룰 tensor 객체는 거의 없다.

data to tensor:

```python
data = [[1, 2], [3, 4]]
t_data = torch.tensor(data)
```

ndarray to tensor:

```python
nd_array_ex = np.array(data)
t_array = torch.from_numpy(nd_array_ex)
```

## Tensor data types

| Data Type                 | dtype                             | CPU tensor             | GPU tensor                  |
|---------------------------|-----------------------------------|------------------------|-----------------------------|
| 32-bit floating point     | `torch.float32` or `torch.float`  | `torch.FloatTensor`    | `torch.cuda.FloatTensor`    |
| 64-bit floating point     | `torch.float32` or `torch.double` | `torch.DoubleTensor`   | `torch.cuda.DoubleTensor`   |
| 16-bit floating point `1` | `torch.float16` or `torch.half`   | `torch.HalfTensor`     | `torch.cuda.HalfTensor`     |
| 16-bit floating point `2` | `torch.bfloat16`                  | `torch.BFloat16Tensor` | `torch.cuda.BFloat16Tensor` |


## Numpy like operations

* 기본적으로 numpy의 대부분의 사용법이 그대로 적용된다
* `.flatten()`, `torch.ones_like()`, `.numpy()`, `.shape()`, `type`
* 단 cpu 또는 gpu 중 어디에 올렸는지 확인하는 `.device` attribute과 `.to('cuda')` 같은 operation이 있다.

```python
data = [[1, 2, 3],[4, 5, 6], [7, 8, 9]]
x_data = pytorch.tensor(data)
x_data[1:]
>>> tensor([[4, 5, 6],
            [7, 8, 9]])

x_data[:2, 1:]
>>> tensor([[2, 3],
            [5, 6]])

x_data.flatten()
>>> tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])

torch.ones_like(x_data)
>>> tensor([[1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]])

x_data.numpy()
>>> array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])

x_data.shape
>>> torch.Size([3, 3])

x_data.device
>>> device(type='cpu')

if torch.cuda.is_available():
    x_data_cuda = x_data.to('cuda')
x_data_cuda.device
>>> device(type='cuda', index=0)
```

## Tensor Handling

- `.view()`, `.squeeze()`, `.unsqueeze()` 등으로 tensor 조정이 가능하다
- `.view()`: `.reshape()`과 동일하게 tensor의 shape을 변환한다
- `.squeeze()`: 차원의 개수가 1인 차원을 삭제 (압축)
- `.unsqueeze()`: 차원의 개수가 1인 차원을 추가

```python
tensor_ex = torch.rand(size = (2, 3, 2))
tensor_ex
>>> tensor([[[0.2707, 0.2995],
             [0.9493, 0.3023],
             [0.2432, 0.2512]],

             [[0.1613, 0.9063],
              [0.0365, 0.3489],
              [0.6682, 0.3997]]])

tensor_ex.view([-1, 6])
>>> tensor([[0.2707, 0.2995, 0.9493, 0.3023, 0.2432, 0.2512],
            [0.1613, 0.9063, 0.0365, 0.3489, 0.6682, 0.3997]])

a = torch.zeros(3, 2)
b = a.view(2, 3)
a.fill_(1)
>>> tensor([[1., 1.],
            [1., 1.],
            [1., 1.]])

a
>>> tensor([[1., 1.],
            [1., 1.],
            [1., 1.]])

b
>>> tensor([[1., 1., 1.],
            [1., 1., 1.]])
```
```python
# 2d tensor
ex = torch.tensor([[1, 2], [3, 4]])
ex.unsqueeze(0)
>>> [[1, 2],
     [3, 4]]       # dim: 1, 2, 2

ex.unsqueeze(1)
>>> [[[1,2]],
     [[3,4]]]   # dim: 2, 1, 2

ex.unsqueeze(2)
>>>  [[[1], [2]],
      [[3], [4]]]   #  dim: 2, 2, 1
```

* 행렬곱셈 연산 함수는 dot이 아닌 `.mm` 사용
* `mm`과 `matmul` 차이: `matmul`은 broadcasting 지원한다

```python
n1 = np.arange(10).reshape(2, 5)
n2 = np.arange(10).reshape(5, 2)
t1 = torch.FloatTensor(n1)
t2 = torch.FloatTensor(n2)

t1.mm(t2)
```
- - -
# Tensor operations for ML and DL formula

- `nn.functional`` 모듈을 통해 다양한 수식 변환을 지원한다
- 그냥 필요할 때 찾아보면 된다.

```python
import torch
import torch.nn.functional as F

tensor = torch.FloatTensor([0.5, 0.7, 0.1])
h_tensor = F.softmax(tensor, dim = 0)

y = torch.randint(5, (10, 5))
y_label = y.argmax(dim = 1)

F.one_hot(y_label)

torch.cartesian_prod(tensor_a, tensor_b)
```

# AutoGrad

* PyTorch의 핵심은 자동 미분의 지원 -> `.backward()` 함수 사용

```python
w = torch.tensor(2.0, requires_grad = True)
y = w ** 2
z = 10 * y + 25
z.backward()
w.grad
>>> tensor(40.)

a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)
Q = 3 * a ** 3 - b ** 2
external_grad = torch.tensor([1., 1.])
Q.backward(gradient = external_grad)
a.grad
>>> tensor([36., 81.])
```

- - -