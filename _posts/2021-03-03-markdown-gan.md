---
title: "Generative Adversarial Networks"
layout: post
category: blog
star: true
use_math: true
author: jiin
---

Goodfellow, Ian J., et al. "Generative adversarial networks." *arXiv preprint arXiv:1406.2661* (2014).

## 목차

1. [Introduction](#1.-introduction)
2. [Related work](#2.-related-work)
3. [Adversarial nets](#3.-adversarial-nets)
4. [Theoretical Results]
5. [Experiments]

<br>

<br>

## 1. Introduction

Deep *generative* model은 다루기 힘든 확률 계산이 많고, [piecewise linear unit](#piecewise-linear-unit)의 장점을 활용하기 어렵기 때문에 지금까지 영향력이 없었다. 그래서 이 어려움들을 피하기 위해 새로운 generative model estimation procedure를 제안한다.  

<br>

Discriminative model은 표본이 model distribution에서 나온 것인지 아니면 data distribution에서 나온 것인지 결정하는 것을 학습한다. 

Generative model은 화폐 위조범과 유사해서 가짜 화페를 만들어서 들키지 않고 사용하려고 하는 반면, discriminative model은 경찰과 유사해서 위조 화폐를 감지하려고 한다. 

이 게임에서 두 팀(화폐 위조범 & 경찰)은 위조 화폐가 진짜 화폐와 구별되지 않을때까지 각자의 방법을 향상시킨다.

<br>

Generative model은 multilayer perceptron 구조로, random noise가 주어지면 samples를 생성한다.

Discriminative model도 역시 multilayer perceptron 구조를 가진다.

Backpropagation과 dropout algorithms를 사용해서 두 개의 모델(generative model & discriminative model)을 훈련시키고, forward propagation을 사용해서 generative model로부터 sample을 생성한다. Approximate inference와 Markov chains는 필요없다.

<br>

<br>

## 2. Related work 

`lack of background knowledge...`

* Restricted Boltzmann machines (RBMs) [27, 16]
* Deep Boltzmann machines (DBMs) [26]
* Markov chain Monte Carlo (MCMC) [3, 5]
* Deep belief networks (DBNs) [16]
* score matching [18]
* Noise-contrastive estimation (NCE) [13]
* Generative stochastic network [5]
* Denoising auto-encoders [4]
* piecewise linear units [19, 9, 10]
* auto-encoding variational Bayes [20]

<br>

<br>

## 3. Adversarial nets

Generative model과 discriminative model이 모두 다중 레이어 퍼셉트론일 때 adversarial modeling framework가 가장 간단하다.

데이터 $x$에 대해 generator의 분포 $p_{g}$를 학습하기 위해, 미리 input noise variables인 $p_{z}(z)$를 정의한다. 

그 후 $G(z;\theta_{g})$로 data space에 대한 mapping을 나타냈고, 거기에서 $G$는 파라미터 $\theta_{g}$와 함께 다중 레이어 퍼셉트론에 의해 나타내지는 미분가능한 함수이다. 

또한 두 번째 다중 레이어 퍼셉트론 $D(z;\theta_{d})$를 정의했고 그것은 하나의 스칼라 값을 출력한다. $D(x)$는 $p_{g}$가 아닌 데이터로부터 $x$가 나올 확률을 나타낸다. 우리는 training samples와 $G$에서 만들어진 samples 모두에게 올바른 레이블을 할당할 확률을 최대화하기 위해 $D$를 훈련시킨다. 우리는 동시에 $log(1-D(G(z))))$을 최소화하기 위해 $G$를 훈련한다. 

------

## 참고

### Piecewise Linear Unit

$PLU(x) == max( alpha (x+c)-c,`min( alpha (x-c)+c,`x))$ -> 수정하기

![PLU](..\assets\gan\plu.PNG)

