---
title: "[boostcamp AI Tech][DL Basic] Lecture 3: Optimization"
date: 2022-10-03 11:00:00 + 0900
categories: [boostcamp AI Tech, Week 3 - DL Basic]
tags: [boostcamp, dl basic, level 1, week 2] # TAG names should always be lowercase
math: true
---

- [Going into the Lecture](#going-into-the-lecture)
- [Important Concepts in Optimization](#important-concepts-in-optimization)
  - [Generalization(일반화)](#generalization일반화)
  - [Underfitting vs. Overfitting](#underfitting-vs-overfitting)
  - [Cross-Validation](#cross-validation)
  - [Bias and Variance](#bias-and-variance)
  - [Bootstrapping](#bootstrapping)
  - [Bagging vs. Boosting](#bagging-vs-boosting)
- [Gradient Descent Methods](#gradient-descent-methods)
  - [Batch-size Matters](#batch-size-matters)
- [Gradient Descent Methods](#gradient-descent-methods-1)
  - [Gradient Descent](#gradient-descent)
  - [Momentum](#momentum)
  - [Nesterov Accelerated Gradient(NAG)](#nesterov-accelerated-gradientnag)
  - [Adagrad](#adagrad)
  - [Adadelta](#adadelta)
  - [RMSprop](#rmsprop)
- [Adaptive Moment Estimation(Adam)](#adaptive-moment-estimationadam)
- [Regularization](#regularization)
  - [Early Stopping](#early-stopping)
  - [Parameter Norm Penalty](#parameter-norm-penalty)
  - [Data Augmentation](#data-augmentation)
  - [Noise Robustness](#noise-robustness)
  - [Label Smoothing](#label-smoothing)
  - [Dropout](#dropout)
  - [Batch Normalization](#batch-normalization)

# Going into the Lecture

- Gradient Descent
  - This is the **1st order iterative optimization algorithm** for finding a local minimum of a differentiable function.

# Important Concepts in Optimization

- Generalization
- Under-fitting vs. Over-fitting
- Cross Validation
- Bias-Variance Tradeoff
- Bootstrapping
- Bagging and Boosting

## Generalization(일반화)

![](/assets/img/boostcamp/2022-10-03-11-30-30.png)

- How well the learned model will behave on unseen data
- If the test error is close to training error, then the model has good generalization.

## Underfitting vs. Overfitting

![](/assets/img/boostcamp/2022-10-03-11-32-37.png)

- Your model is *underfitting* the training data when the model performs poorply on the training data
- Your model is *overfitting* your training data if you see that the model performs well on the training data but does not perform well on the test data
  - This is because the model is memorizing the data it has seen and is unable to generalize to unseen samples.

## Cross-Validation

![](/assets/img/boostcamp/2022-10-03-11-36-22.png)

- *Cross-Validation* is a model validation technique for assessing how the model will generalize to an independent (test) data set.
- It is also called *K-Fold Validation*

## Bias and Variance

![](/assets/img/boostcamp/2022-10-03-11-41-40.png)

- Variance: How close are the outputs given similar inputs? Low Variance = outputs close together
- Bias: How close is the output to the target? Low Bias = output close to target

## Bootstrapping

![](/assets/img/boostcamp/2022-10-03-11-43-34.png)

- Bootstrapping in any test or metric that uses random sampling with replacement
- And then we check the consensus of the models to test *uncertainty*
  
## Bagging vs. Boosting

![](/assets/img/boostcamp/2022-10-03-11-48-46.png)

- Bagging is short for Bootstrapping aggregating
  - Multiple models are being trained with bootstrapping
  - ex) Base classifiers are fitted on random subset where indivdual predictions are aggregated (voting or averaging)
  - Often use Ensemble technique
- Boosting
- It focuses on those specific training samples that are hard to classify
- A strong model is built by combining weak learners in sequence where each leaerner learns from the mistakes of the previous weak learner


# Gradient Descent Methods

- Stochastic Gradient Descent (SGD)
  - Update with the gradient computed from a single sample
- Mini-batch Gradient Descent
  - Update with the gradient computed from a subset of data
- Batch Gradient Descent
  - Update with the gradient computed from the whole data.

## Batch-size Matters

![](/assets/img/boostcamp/2022-10-03-11-53-09.png)

- Large batch methods tend to converge to *sharp minimizers* of the training and testing functions
- Small batch methods consistently converge to *flat minimizers*
- This is due to the inherent noise in the gradient estimation
- Generalization Performace improves with smaller batch methods.

# Gradient Descent Methods

- Stochastic Gradient Descent
- Momentum
- Nesterov Accelerated Gradient
- Adagrad
- Adadelta
- RMSprop
- Adam

## Gradient Descent

$$
  W_{t+1} \leftarrow W_{t} - \eta g_{t}
$$

where $W = weight$, $g_t = gradient$ and $\eta = learning \space rate$

- It's important to choose the appropriate learning rate

## Momentum

$$
\begin{aligned}
  & a_{t+1} \leftarrow \beta a_{t} + g_{t} \\
  & W_{t+1} \leftarrow W_{t} - \eta a_{t+1}
\end{aligned}
$$

where $a_{t+1} = accumulation$ and $\beta = momentum$

- $\beta a_{t}$ acts as a stabilizer pushing the accumulation to one direction
- helps to overcome oscillation

## Nesterov Accelerated Gradient(NAG)

![](/assets/img/boostcamp/2022-10-03-13-13-46.png)

- NAG is a slightly modified version of Momentum with stronger theoretical convergence guarantees for convex functions
- In practice, it has produced slightly better results than classical Momentum
- [Further Reading](https://golden.com/wiki/Nesterov_momentum)

$$
\begin{aligned}
   a_{t+1} &\leftarrow \beta a_{t} + \nabla\mathcal{L}(W_{t} - \eta\beta a_{t}) \\
    W_{t+1} &\leftarrow W_{t} - \eta a_{t+1}
\end{aligned}
$$

- $\nabla\mathcal{L}(W_{t} - \eta\beta a_{t})$ is the *lookahead gradient* and it tries to prevent overshoot.

## Adagrad

$$
\begin{aligned}
  W_{t+1} = W_t - \frac{\eta}{\sqrt{G_{t} + \epsilon}} g_{t}
\end{aligned}
$$

where $G_{t}$ = Sum of gradient squares and $\epsilon$  is for numerical stability.

- Adagrad adapts the learning rate, performing larger updates for infrequent parameters and smaller updates for frequent parameters
- Ada is short for adaptive learning rate approach
- If training occurs for a long period, $\frac{\eta}{\sqrt{G_{t} + \epsilon}}$ converges to 0 resulting in no learning

## Adadelta

- Adadelta extends Adagrad to reduce its monotonically decreasing the learning rate by restricting the accumulation window.

$$
\begin{aligned}
  G_t &= \gamma G_{t-1} + (1-\gamma)g_{t}^2 \\
  W_{t+1} &= W_{t} - \frac{\sqrt{H_{t-1} + \epsilon}}{\sqrt{G_{t}+\epsilon}}g_t \\
  H_t &= \gamma H_{t-1} + (1-\gamma)(\Delta W_t)^2
\end{aligned}
$$

where $G_t$ = EMA of gradient squares and $H_t$ = EMA of difference squares (EMA = Exponential Moving Average)
  
- There is no learning rate in Adadelta
- We don't really use this a lot

## RMSprop

- RMSprop is an unpublished, adaptive learning rate method proposed by Geoff Hinton in his lecture.

$$
\begin{aligned}
  G_t &= \gamma G_{t-1} + (1-\gamma)g_{t}^2 \\
  W_{t+1} &= W_{t} - \frac{\eta}{\sqrt{G_{t}+\epsilon}}g_t
\end{aligned}
$$

where $\eta$ = stepsize

- RMSprop adds stepsize to Adadelta

# Adaptive Moment Estimation(Adam)

- Adam leverges both past gradients and squared gradients.
- So combined the ideas in momentum and adaptive learning rate approach.

$$
\begin{aligned}
    m_t &= \beta_1 m_{t=1} + (1-\beta_1)g_t \\
    v_t &= \beta_@ v_{t-1} + (1-\beta_2)g_t^2 \\
    W_{t+1} &= W_t - \frac{\eta}{\sqrt{v_t + \epsilon}}\frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t}m_t
\end{aligned}
$$

where $m_t$ = momentum, $v_t$ = EMA of gradient squares and $\frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t}$ = unbiased estimator.

- Best to use. 

# Regularization

- Tools used to make better generalizations
- Early stopping
- Parameter norm penalty
- Data augmentation
- Noise robustness
- Label smoothing
- Dropout
- Batch normalization

## Early Stopping

![](/assets/img/boostcamp/2022-10-03-13-32-51.png)

- Stop iteration early when the generalization gap increases.

## Parameter Norm Penalty

- It adds smoothness to the function space.

$$
\text{total cost} = \text{loss}(\mathcal{D}; W) + \frac{\alpha}{2}||W||_2^2
$$

- also called weight decay
- prevents parameter from exploding by also minimizing the Parameter Norm Penalty, $\frac{\alpha}{2}||W||_2^2$

## Data Augmentation

- More data the better
- Artificially increase the number of data by undergoing label preserving augmentations (strech, rotate, etc)

## Noise Robustness

![](/assets/img/boostcamp/2022-10-03-13-39-31.png)

- Add random noise to inputs or weights
- It somehow works better.

## Label Smoothing

- **Mix-up** constructs augmented training examples by mixing both input and output of two randomly selected training data
- **CutMix** constructs augmented training exmaples by mixing inputs with cut and paste and outputs with soft labels of two randomly selected training data
- **CutOut** crop a certain section of training examples.

## Dropout

- In each forward pass, randomly set some neurons to zero.'
- It's like training multiple models.

## Batch Normalization

- Batch normalization compute the empirical mean and variance independently for each dimension (layers) and normalize.

$$
\begin{aligned}
  \mu_{B} &= \frac{1}{m}\sum_{i=1}^m x_i \\
  \sigma^2_{B} &= \frac{1}{m}\sum_{i=1}^m (x_i - \mu_{B})^2 \\
   \hat{x}_i &= \frac{x_i - \mu_{B}}{\sqrt{\sigma^2_{B}+\epsilon}}
\end{aligned}
$$

- There are other norms like Layer Norm, Instance Norm and Group Norm

![](/assets/img/boostcamp/2022-10-03-13-45-04.png)

-----------------------------------


