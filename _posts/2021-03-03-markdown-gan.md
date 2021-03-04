---
title: "Generative Adversarial Networks"
layout: post
category: blog
star: true
use_math: true
author: jiin
---

Goodfellow, Ian J., et al. "Generative adversarial networks." *arXiv preprint arXiv:1406.2661* (2014).

[TOC]

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

데이터 $x$에 대해 generator의 분포 $p_g$를 학습하기 위해, 미리 input noise variables인 $p_z(z)$를 정의한다. 

그 후 $G(z;\theta_g)$로 data space에 대한 mapping을 나타냈고, 거기에서 $G$는 파라미터 $\theta_g$와 함께 다중 레이어 퍼셉트론에 의해 나타내지는 미분가능한 함수이다. 

또한 1개의 스칼라 값을 출력하는 두 번째 다중 레이어 퍼셉트론 $D(z;\theta_d)$를 정의했다. $D(x)$는 $p_g$가 아닌 데이터로부터 $x$가 나올 확률을 나타낸다. 우리는 training samples와 $G$에서 만들어진 samples 모두에게 올바른 레이블을 할당할 확률을 최대화하기 위해 $D$를 훈련시킨다. 동시에 $log(1-D(G(z))))$을 최소화하기 위해 $G$를 훈련한다:

즉, $D$와 $G$는 value function $V(G, D)$와 함께 two-player minimax game을 한다.



Eq 1.
$$
\min_{G}\max_{D}V(D, G)=\mathbb{E}_{x\sim p_{data}}[logD(x)]+\mathbb{E}_{z\sim p_z(z)}[log(1-D(G(z)))]
$$
<br>

다음 섹션에서, 우리는 adversarial nets의 이론적인 분석을 제시하고, 본질적으로 training criterion이 $G$가 data generating distribution을 복구하고 $D$에는 충분한 용량이 주어진다, 즉, non-parametric limit. 

Figure 1을 보자.  훈련의 내부 루프에서 완료될 때까지 $D$를 최적화 하는 것은 계산적으로 금지되어 있고, 유한한 데이터셋은 overfitting을 발생시킬 것이다. 대신에, 우리는 교대로 $k$번 $D$를 최적화하고 $1$번 $G$를 최적화한다. $D$에서의 결과가 최적의 결과 가까이 유지되고, $G$가 충분히 천천히 변화할때까지 유지된다. 

이 전략은 SML/PCD [31, 29]와 유사하다. 훈련이 Markov chain으로부터 samples를 유지한다. 한 개의 learning step에서부터 다음 step까지. Markov chain에서 끝나는 것을 피하기 위해. 학습의 내부 루프의 일부로써. 절차는 공식적으로 Algorithm 1에서 제시된다.

<br>

실제로, 식 1은 잘 학습하기 위한 $G$에 충분한 기울기를 제공하지 않는다. 학습의 초기에, $G$가 가난할 때, $D$는 높은 확신을 가지고 samples를 거절할 수 있다. 왜냐하면 그들은 명백히 훈련 데이터와 다르기 때문이다. 이 경우, $log(1-D(G(z)))$는 포화된다. $log(1-D(G(z)))$를 최소화시키기 위해 $G$를 훈련시키는 것보다 우리는 $G$를 $log D(G(z))$를 최대화시키는 것으로 훈련할 수 있다. 이 목적 함수는 같은 $G$와 $D$의 같은 fixed point of the dynamics를 결과로 내지만 훈련 초기에 더 강력한 기울기를 제공한다.

<br>

![Fig1](C:\Users\jiinkim\Desktop\ji-in.github.io\assets\gan\fig1.PNG)

Figure 1: Generative adversarial nets는 동시에 discriminative distribution ($D$, blue, dashed line)을 동시에 업데이트 시키면서 훈련하고 그래서 이것은 data generating distribution (black, dotted line)으로부터의 samples $p_x$를 generative distribution $p_g$ (G) (green, solid line)로부터 구별한다. 

아래의 수평선은 $z$가 sampled되는 domain이고, 이 경우 uniformly하다. 위쪽 수평선은 $x$의 일부이다. 

위를 향하는 화살표는 mapping $x=G(z)$가 어떻게 non-uniform distribution $p_g$를 transformed samples에 도임하는지 보여준다. $G$는 높은 밀도를 가진 부분에서 줄어들고 낮은 밀도를 가진 $p_g$ 부부에서 팽창한다. 

(a) 수렴 가까이에 있는 adversarial pair를 고려해보자: $p_g$는 $p_data$와 유사하고 $D$는 부분적으로 정확한 classifier이다. 

(b) 알고리즘의 내부 루프에서 $D$는 데이터로부터 samples를 식별하기 위해 훈련되고, 수렴하면서 $D^{*}(x)=\frac{p_{data}(x)}{p_{data}(x)+p_{g}(x))}$.

(c) $G$를 갱신한 후에, $D$의 기울기는 $G(z)$가 data로 분류될 가능성이 더 높은 영역으로 흐르도록 안내했다.

(d) 훈련의 몇 단계를 거친 후, 만약 $G$와 $D$가 충분히 capacity가 있다면, 그들은 둘 다 개선할 수 없는 지점에 도달할 것이다. 왜냐하면 $p_g=p_data$.

discriminator는 두 개의 분포 사이에서 미분할 수 없다, 즉, $D(x)=\frac{1}{2}$

------

## 참고

### Piecewise Linear Unit

$PLU(x) == max( alpha (x+c)-c,`min( alpha (x-c)+c,`x))$ -> 수정하기

![PLU](..\assets\gan\plu.PNG)

