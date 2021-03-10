---
title: "[PGGAN] Progressive Growing of GANs for Improved Quality, Stability, and Variation"
layout: post
category: blog
use_math: true
author: jiin
---

Paper: [https://arxiv.org/abs/1710.10196](https://arxiv.org/abs/1710.10196)



## 1. Introduction

가장 유명한 generative methods는 Autoregressive models, VAEs, GANs가 있다.

Autoregressive models는 sharp image를 생성하지만 evaluation이 느리고 latent representation을 가지지 않는다.

VAEs는 train은 쉽지만 blurry한 결과를 출력한다.

GANs도 sharp image를 생성하지만, small resolution(저해상도)에서 동작하고 variation이 제한적이며 training이 불안정하다. 

<br>

High resolution(고해상도) 이미지에서 generated image와 training image는 너무 쉽게 구별이 간다. 화질이 높기 때문에 아주 작은 차이도 잡아낼 수 있기 때문이다. 그래서 high resolution에서의 이미지 생성은 어렵다. 

또한 large resolution은 메모리 제한때문에 더 작은 minibatch를 사용해야 한다.

Key insight는 generator와 discriminator를 점진적으로 키우는 것이다. low-resolution 이미지부터 시작하고 학습이 진행될 때 higher-resolution details를 사용하는 새로운 레이어를 추가한다.

![Output](..\assets\pggan\output2.gif)

<br>

## 2. Progressive Growing of GANs

![Network](..\assets\pggan\structure.PNG)

Training은 저해상도 4 x 4 이미지와 함께 generator(G)와 discriminator(D)에서 시작한다. 학습이 진행되면, 점진적으로 G와 D에 레이어를 추가해서 생성 이미지의 spatial resolution을 증가시킨다. 

이 방법은 안정적으로 high resolution 이미지를 만들고, 학습 속도를 상당히 빠르게 한다.

<br>

학습을 할 때 네트워크를 점진적으로 키우면, 이미지에서 전체적인 구조부터 찾고 점차 더 세밀한 곳으로 집중하게 된다.

Generator와 discriminator의 구조는 서로 대칭적이고 항상 동시에 커진다. 학습을 하는 동안, 두 네트워크의 모든 레이어들은 계속 trainable하다. 네트워크에 새로운 레이어가 추가되면, `fade in`을 사용한다.

![Fade in](..\assets\pggan\fade_in.PNG)

(a)에서 (c)로 가기 위해 (b) 과정을 거친다.

`2x`는 nearest neighbor filtering을 사용해서 image resolution을 두 배로 만들고, `0.5x`는 average pooling을 사용해서 image resolution을 반으로 줄인다. `toRGB`는 feature vectors를 RGB colors로 바꾸는 과정이고, `fromRGB`는 그 반대의 과정이다.

16 x 16 해상도 이미지를 만든 후, 바로 32 x 32 해상도를 학습시키면 아직 학습이 안된 32 x 32 레이어가 잘 학습된 16 x 16 레이어에 안좋은 영향을 끼칠 수 있다. 

그래서 `fade in smoothly`가 필요하다.

G에서의 동작은 다음과 같다. 

(b)에서 16 x 16 해상도 이미지 크기를 2배로 늘려 크기만 32 x 32인 이미지를 만든다. 

32 x 32 레이어의 결과 이미지인 고해상도 이미지의 각 픽셀 값에 0~1 사이의 비율을 나타내는 $\alpha$를 곱하고, 저 해상도 픽셀에는 그 나머지 비율인 $1-\alpha$을 곱한다. 

두 개의 결과를 서로 더하면 저해상도의 큰 그림과 앞으로 학습시킬 고해상도의 디테일이 합쳐진다. 

D에서도 위와 같은 방식으로 진행한다. 

`fade in`을 사용하면, 기존의 저해상도 레이어에서 학습한 것을 해치지 않고, 새로 추가된 레이어를 학습시킬 수 있다. 

<br>

## 3. Increasing Variation using Minibatch standard deviation

`minibatch discrimination`을 사용한다.



## 4. Normalization in Generator and Discriminator

### 4.1. Equalized Learning Rate

### 4.2. Pixelwise Feature Vector Normalization in Generator

------

Reference

[https://towardsdatascience.com/progan-how-nvidia-generated-images-of-unprecedented-quality-51c98ec2cbd2](https://towardsdatascience.com/progan-how-nvidia-generated-images-of-unprecedented-quality-51c98ec2cbd2)

[https://wiserloner.tistory.com/1196](https://wiserloner.tistory.com/1196)

[https://sensibilityit.tistory.com/508?category=731657](https://sensibilityit.tistory.com/508?category=731657)

