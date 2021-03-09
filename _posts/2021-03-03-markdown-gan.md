---
title: "Generative Adversarial Networks"
layout: post
category: blog
star: true
use_math: true
author: jiin
---

**Goodfellow, Ian J., et al. "Generative adversarial networks." *arXiv preprint arXiv:1406.2661* (2014).**

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

Adversarial modeling framework는 generative model과 discriminative model이 모두 다중 레이어 퍼셉트론일 때 가장 간단하다. 

1. Input noise variables인 $p_z(z)$를 미리 정의한다. 

2. 다중 레이어 퍼셉트론인 $G(z;\theta_g)$를 정의한다. 

3. 다중 레이어 퍼셉트론인 $D(z;\theta_d)$를 정의한다. $D(x)$는 데이터로부터 $x$가 나올 확률을 나타낸다. 

4. $D$는 Training samples와 $G$가 생성한 samples를 구별하는 것을 학습한다. 동시에 $log(1-D(G(z))))$을 최소화하기 위해 $G$가 학습한다: 

<br> 

<img src="C:\Users\jiinkim\Desktop\ji-in.github.io\assets\gan\eq1.PNG" style="zoom:80%;" />

<br>

Figure 1을 보자.  

우리는 교대로 $k$번 $D$를 최적화하고 $1$번 $G$를 최적화한다. $G$가 충분히 천천히 변화할때까지 $D$에서 최적의 결과가 유지된다. 

<br>

실제로, 식 1은 잘 학습하기 위한 $G$에 충분한 기울기를 제공하지 않는다. 학습 초기에는 $G$가 명백히 훈련 데이터와 다른 결과를 생성하기 때문에, $D$는 확신을 가지고 $G$가 생성한 이미지를 가짜로 분류한다. 이 경우, $log(1-D(G(z)))$는 포화된다. 

$G$를 $log(1-D(G(z)))$를 최소화시키기 위해 학습하는 것보다 $log D(G(z))$를 최대화시키는 것으로 학습할 수 있다. $log D(G(z))$는 훈련 초기에 더 강력한 기울기를 제공한다.

<br>

![Figure1](..\assets\gan\fig1.PNG)

Figure 1:

- Data generating distribution : black, dotted line
- <span style="color:blue">Discriminative distribution : blue, dashed line</span>
- <span style="color:green">Generative distribution : green, solid line</span>

Discriminative distribution는 갱신과 훈련을 동시에 하고, $p_x$를 $p_g$로부터 구별한다.

아래에 있는 수평선은 $z$가 sampled되는 domain이고, 위에 있는 수평선은 $x$의 일부이다. 

화살표가 몰려있는 부분에서 $G$는 아래로 떨어지는 모양이고, 그렇지 않은 부분에서 $G$는 옆으로 퍼지는 모양이다.

**각각의 그림들을 개별적으로 설명해보자**

- (a) 수렴하는 부분에서 $p_g$는 $p_data$와 유사하다.
- (b) $D$는 데이터로부터 samples를 구별하기 위해 훈련되고, $D^{*}(x)=\frac{p_{data}(x)}{p_{data}(x)+p_{g}(x))}$로 수렴한다. (수렴하면 $\frac{1}{2}$이 된다)
- (c) $G$를 갱신한 후에, $D$의 기울기는 $G(z)$가 data로 분류될 가능성이 더 높은 영역으로 갈 수 있도록 안내한다.
- (d) 훈련을 몇 번 한 후, 만약 $G$와 $D$가 충분히 capacity가 많다면(학습 파라미터의 수가 많다면), $G$와 $D$는 $p_g=p_data$이기 때문에 포화될 것이다. Discriminator는 $D(x)=\frac{1}{2}$가 된다.

<br>

<br>

## 4. Theoretical Results

Generator $G$는 암묵적으로 probability distribution $p_g$로 정의된다. $z\sim p_z$일 때 distribution of the samples $G(z)$가 얻어지기 때문이다. 그러므로, 우리는 충분한 capacity와 훈련 시간이 주어진다면, Algorithm 1이 $p_data$의 좋은 estimator로 수렴되길 원한다. 이 섹션의 결과는 파라미터가 없는 환경에서 수행된다. 예를 들어, 우리는 무한한 capacity와 함께 모델을 나타낸다. probability density functions의 공간에서 수렴하는 것을 학습함으로써.

<br>

우리는 section 3.1에서 이 minimax game이 $p_g=p_data$에 대해 global optimum을 가지는 것을 보여줄 것이다. 우리는 그런 후 section 4.2에서 Algorithm 1이 Eq 1을 최적화하는 것을 보여주고, 그러므로 바라는 결과를 얻는다.

<br>

<img src="..\assets\gan\algorithm1.PNG" style="zoom:80%;" />



## 4.1 Global Optimality of  $p_g=p_data$

우리는 먼저 어느 주어진 generator $G$를 위해 optimal discriminator $D$를 고려한다.

**명제 1.** $G$가 고정된 경우, optimal discriminator $D$는 다음과 같다.

<img src="C:\Users\jiinkim\Desktop\ji-in.github.io\assets\gan\eq2.PNG" style="zoom:70%;" />

**증명.**

아무 generator $G$가 주어졌을 때 discriminator $D$에 대한 training criterion은 quantity $V(G, D)$를 최대화한다.

<img src="C:\Users\jiinkim\Desktop\ji-in.github.io\assets\gan\eq3.PNG" style="zoom:70%;" />

$(a,b)\in \mathbb{R}^2 \setminus \left\{0, 0\right \}$에 대해서, $y\rightarrow a log(y)+b log(1-y)$는 $\frac{a}{a+b}$에서 $\left [ 0, 1 \right ]$에서 그것의 maximum을 가진다. discriminator는 $Supp(p_{data})\cup Supp(p_g)$의 밖에서 정의될 필요가 없다. 증명이 끝났다.

<br>

$D$를 위한 목적 함수를 training하는 것은 estimating the conditional probability $P(Y=y|x)$를 위해 log-likelihood를 최대화 시키는 것으로 해석되고, $Y$는 $x$가 $p_data$로부터 왔는지 (with $y=1$) $p_g$로부터 왔는지 (with $y=0$)를 나타낸다. Eq. 1에 있는 minimax game은 다음과 같이 다시 정의된다:

<img src="C:\Users\jiinkim\Desktop\ji-in.github.io\assets\gan\eq4.PNG" style="zoom:70%;" />

<br>

**이론 1.** virtual training criterion $C(G)$의 global minimum은  $p_g=p_data$와 필요충분조건일 때 달성된다. 그 점에서 $C(G)$는 $-log4$의 값을 가진다.

**증명.**

$p_g=p_data$에 대해서, D^{*}_G(x)=\frac{1}{2} (Eq. 2를 고려해보아라).

------

## 참고

### Piecewise Linear Unit

$PLU(x)\equiv max(\alpha (x+c)-c, min(\alpha (x-c)+c, x))$

![PLU](..\assets\gan\plu.PNG)

