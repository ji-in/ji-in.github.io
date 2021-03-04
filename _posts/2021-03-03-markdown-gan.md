---
title: "Generative Adversarial Networks"
layout: post
category: blog
author: jiin
---

Goodfellow, Ian J., et al. "Generative adversarial networks." *arXiv preprint arXiv:1406.2661* (2014).

## 목차

1. Introduction
2. Related work
3. Adversarial nets
4. Theoretical Results
5. Experiments

## 

## 1. Introduction

Deep *generative* model은 다루기 힘든 확률 계산이 많고, [piecewise linear unit](#piecewise-linear-unit)의 장점을 활용하기 어렵기 때문에 지금까지 영향력이 없었다. 그래서 이 어려움들을 피하기 위해 새로운 generative model estimation procedure를 제안한다.  

<br>

Discriminative model은 표본이 model distribution에서 나온 것인지 아니면 data distribution에서 나온 것인지 결정하는 것을 학습한다. 

Generative model은 화폐 위조범과 유사해서 가짜 화페를 만들어서 들키지 않고 사용하려고 하는 반면, discriminative model은 경찰과 유사해서 위조 화폐를 감지하려고 한다. 

이 게임에서 두 팀(화폐 위조범 & 경찰)은 위조 화폐가 진짜 화폐와 구별되지 않을때까지 각자의 방법을 향상시킨다.

<br>

Generative model은 multilayer perceptron 구조로, random noise가 주어지면 samples를 생성한다.

Discriminative model도 역시 multilayer perceptron 구조를 가진다.

backpropagation과 dropout algorithms를 사용해서 두 개의 모델(generative model & discriminative model)을 훈련시키고, forward propagation을 사용해서 generative model로부터 sample을 생성한다. Approximate inference와 Markov chains는 필요없다.

<br>

<br>

## 2. Related work



------

## 참고

### Piecewise Linear Unit

$PLU(x) == max( alpha (x+c)-c,`min( alpha (x-c)+c,`x))$

![PLU](..\assets\gan\plu.PNG)

