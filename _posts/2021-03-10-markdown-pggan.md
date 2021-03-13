---
title: "[PGGAN] Progressive Growing of GANs for Improved Quality, Stability, and Variation"
layout: post
category: blog
use_math: true
author: jiin
---

Paper: [https://arxiv.org/abs/1710.10196](https://arxiv.org/abs/1710.10196)

<br>

## 1. Introduction

가장 유명한 generative methods는 Autoregressive models, VAEs, GANs가 있다.

Autoregressive models는 sharp image를 생성하지만 evaluation이 느리고 latent representation을 가지지 않는다.

VAEs는 train은 쉽지만 blurry한 결과를 출력한다.

GANs도 sharp image를 생성하지만, small resolution(저해상도)에서 동작하고 variation이 제한적이며 training이 불안정하다. 

<br>

High resolution(고해상도) 이미지에서 generated image와 training image는 너무 쉽게 구별이 간다. 화질이 높기 때문에 아주 작은 차이도 잡아낼 수 있기 때문이다. 그래서 high resolution에서의 이미지 생성은 어렵다. 

또한 large resolution은 메모리 제한때문에 더 작은 minibatch를 사용해야 한다.

Key insight는 generator와 discriminator를 점진적으로 키우는 것이다. low-resolution 이미지부터 시작하고 학습이 진행될 때 higher-resolution details를 사용하는 새로운 레이어를 추가한다.

<p align="center">
    <img src="..\assets\pggan\output2.gif" alt="output" style="zoom:80%;" />
    Output
</p>



<br>

## 2. Progressive Growing of GANs

<p align="center">
    <img src="..\assets\pggan\structure.PNG" alt="output" style="zoom:80%;" />
    Figure 1
</p>

Training은 저해상도 4 x 4 이미지와 함께 generator(G)와 discriminator(D)에서 시작한다. 학습이 진행되면, 점진적으로 G와 D에 레이어를 추가해서 생성 이미지의 spatial resolution을 증가시킨다. 

이 방법은 안정적으로 high resolution 이미지를 만들고, 학습 속도를 상당히 빠르게 한다.

<br>

학습을 할 때 네트워크를 점진적으로 키우면, 이미지에서 전체적인 구조부터 찾고 점차 더 세밀한 곳으로 집중하게 된다.

<p align="center">
    <img src="..\assets\pggan\output1.gif" style="zoom:80%;" />
    visual representation
</p>



Generator와 discriminator의 구조는 서로 대칭적이고 항상 동시에 커진다. 학습을 하는 동안, 두 네트워크의 모든 레이어들은 계속 trainable하다. 네트워크에 새로운 레이어가 추가되면, `fade in`을 사용한다.

<p align="center">
    <img src="..\assets\pggan\fade_in.PNG" alt="output" />
    Figure 2
</p>

(a)에서 (c)로 가기 위해 (b) 과정을 거친다.

`2x`는 nearest neighbor filtering을 사용해서 image resolution을 두 배로 만들고, `0.5x`는 average pooling을 사용해서 image resolution을 반으로 줄인다. `toRGB`는 feature vectors를 RGB colors로 바꾸는 과정이고, `fromRGB`는 그 반대의 과정이다.

16 x 16 해상도 이미지를 만든 후, 바로 32 x 32 해상도를 학습시키면 아직 학습이 안된 32 x 32 레이어가 잘 학습된 16 x 16 레이어에 안좋은 영향을 끼칠 수 있다. 

그래서 "fade in smoothly"가 필요하다.

G에서의 동작은 다음과 같다. 

(b)에서 16 x 16 해상도 이미지 크기를 2배로 늘려 크기만 32 x 32인 이미지를 만든다. 

32 x 32 레이어의 결과 이미지인 고해상도 이미지의 각 픽셀 값에 0~1 사이의 비율을 나타내는 $\alpha$를 곱하고, 저 해상도 픽셀에는 그 나머지 비율인 $1-\alpha$를 곱한다. 

두 개의 결과를 서로 더하면 저해상도의 큰 그림과 앞으로 학습시킬 고해상도의 디테일이 합쳐진다. 

D에서도 위와 같은 방식으로 진행한다. 

"fade in"을 사용하면, 기존의 저해상도 레이어에서 학습한 것을 해치지 않고, 새로 추가된 레이어를 학습시킬 수 있다. 

<br>

<p align="center">
    <img src="..\assets\pggan\generator.png" />
    Generator
</p>

<p align="center">
    <img src="..\assets\pggan\discriminator.png" />
    Discriminator
</p>
<br>

## 3. Increasing Variation using Minibatch Standard Deviation

GAN은 training 중 train data에서 찾은 feature information보다 variation이 적은 image를 생성하는 경향이 있다. 그로 인해 고해상도 이미지를 생성하기 어렵다는 단점이 있다. Salimans et al. (2016)은 "minibatch discrimination"을 해결책으로 제시했다. 

우리는 "minibatch discrimination"을 단순화한 minibatch standard deviation을 사용해서 variation을 증가시킨다

Minibatch standard deviation은 feature statistics를 이미지 한장에 대해서 계산하는 것 뿐만 아니라, minibatch 전체에서 계산한다. 그래서 generated images의 minibatch와 training images의 minibatch에서 statistics가 비슷해지도록 한다.

이것을 구현하기 위해 discriminator의 끝에 minibatch layer를 추가한다. 

Minibatch의 각 example에 대해 statistics가 만들어지고, 그것을 레이어의 출력에 연결하면, discriminator가 statistics를 내부적으로 사용할 수 있다.

<br>

계산 과정은 다음과 같다. 

* Minibatch 내의 각 spatial location에서 각 feature의 표준편차를 계산한다. 

* 모든 features와 spatial locations에서 계산한 값(표준편차)을 평균 낸다. 

* 값을 복사해서 모든 spatial locations과 minibatch에 연결해서, 한 개의 추가적인 feature map을 생성한다. 

이 레이어는 discriminator의 어디든지 삽입할 수 있지만, 끝에 삽입하는 것이 가장 좋다. 

<br>

## 4. Normalization in Generator and Discriminator

### 4.1. Equalized Learning Rate

Gaussian distribution을 사용해서 가중치를 초기화 하고, runtime 동안에 가중치를 조정한다.

$\hat{w}_i=w_i/c$를 사용한다. $w_i$는 가중치이고, $c$는 He 초기화의 per-layer 정규화 상수이다.

다른 방법들은 보통 파라미터의 scale과 무관하게 gradient를 갱신한다. 이때 파라미터마다 dynamic range가 다르면 이 값을 조절하는 데에 시간이 많이 걸린다.

그러나 equalized learning rate를 사용하면, 모든 가중치들이 동일한 dynamic range를 가짐으로써 동일한 learning speed를 가지는 것을 보장한다.

<br>

### 4.2. Pixelwise Feature Vector Normalization in Generator

각 합성곱 연산 후에 generator에서 각 픽셀의 feature vector를 단위 길이로 정규화한다. 

"local response normalization"의 변형을 사용하여 이것을 수행한다.

<p align="center">
    <img src="..\assets\pggan\norm.PNG" style="zoom:80%;" />
</p>

$N$: feature map의 수

$a_x,y$: pixel $(x, y)$에서의 original feature vector

$b_x,y$: pixel $(x, y)$에서의 normalized feature vector

이 방법은 결과를 바꾸지는 않지만, training 중에 signal의 크기가 갑자기 커지는 현상을 막아준다.

<br>

## 5. Multi-Scale Statistical Similarity for Assessing GAN Results

GAN의 결과들을 비교하기 위해, 수많은 이미지들을 사람이 일일이 비교해야 한다. 이 작업은 힘들기 때문에 자동화된 방법을 쓴다. 

MS-SSIM과 같은 기존 방법은 대량의 mode collapses를 잘 찾지만 색상 변화나 질감 변화같은 작은 효과에는 반응하지 않는다. 또한 training set와의 유사성을 평가하지 않는다.

잘 만든 generator의 이미지는 어떤 스케일이든 상관없이 local한 부분이 training set과 유사할 것이다. Generated images(low resolution  16 x 16)와 target images(low resolution 16 x 16)를 Laplacian pyramid에 넣고, 출력으로 나온 local image patch들의 분포 사이에서 multi-scale statistical similarity를 구하는 방법을 제안한다.

Training set과 generated set으로부터 각각 local image patch를 얻는다. 각 color channel의 평균과 표준편차로 patch들을 정규화한다. 이후 sliced Wasserstein distence (SWD)를 통해 statistical similarity를 계산한다. `(완벽 이해 x)`

작은 Wasserstein distance는 patch들 간의 분포가 유사하다는 것을 나타내고, training images와 generator samples가  spatial resolution에서 apperance와 variation가 유사하다는 것을 의미한다. 

Lowset resolution 16 x 16 images로부터 추출된 patch sets 사이의 거리는 large-scale image structures에서 유사성을 나타내고, Finest-level patches는 edge와 noise의 sharpness와 같은pixel-level attrubutes에 대한 정보를 encode한다.

<br>

## 6. Experiments

### 6.1. Importance of Individual Contributions in Teams of Statistical Similarity

Contribution을 평가하기 위해 sliced Wasserstein distance (SWD)와 multi-scale structural similarity (MS-SSIM)을 사용한다. Loss는 WGAN-GP를, 데이터셋은 CelebA와 LUSN bedroom을 사용한다.

<p align="center">
    <img src="..\assets\pggan\table1.PNG" style="zoom:80%;" />
    Table 1
</p>

SWD와 MS-SSIM을 계산한 결과이다.

SWD에서, 각 열은 Laplacian pyramid의 level을 나타내고, 마지막 열은 네 개 distance의 평균값이다.

<p align="center">
    <img src="..\assets\pggan\figure3.PNG" />
    Figure 3
</p>
**(a) - (g)** 는 Table 1의 행에 해당하는 CelebA 예제들이다. 이것들은 내부적으로 수렴하지 않는다. 

**(h)**는 우리의 수렴된 결과이다.

<br>

좋은 평가 지표는 색상, 질감, 방향에서 많은 variation이 있는 그럴듯한 이미지인지 판단할 수 있어야 한다.

MS-SSIM은 outputs 사이에서의 variation만을 측정하기 때문에 generated images와 training set의 유사성을 판단할 수 없다. 그러나, SWD는 generated images의 분포가 training set와 유사하다는 것을 올바르게 찾는다.

첫 번째 configuration **(a)**는 Gulrajani et al. (2017)이다. Generator에서 batch normalization을 사용하고 Discriminator에서 layer normalization을 사용하며, minibatch 크기는 64이다. 

**(b)**는 네트워크의 progressive growing을 추가했고, 더 좋은 결과를 낸다. 

주요한 목표는 고해상도 이미지를 만드는 것이다. 고해상도 이미지를 만드려면 메모리 제한으로 인해 mini-batches의 크기를 줄여야 한다. **(c)**에서 minibatch 크기를 64에서 16으로 줄였다. 그 때 생성된 이미지는 상당히 부자연스럽다. MS-SSIM과 SWD에서도 결과가 좋지 않은 것을 확인할 수 있다.

**(d)**는 hyperparameters를 조정해서 training process를 안정화시키고, batch normalization과 layer normalization을 제거했다.

**(e*)**는 minibatch discrimination을 사용하지만, 그다지 평가 지표의 값들을 개선하지 못한다.

대조적으로 우리의 minibatch standard deviation **(e)**는 average SWD와 이미지들을 개선한다.

그런 후, **(f)**와 **(g)**에서 우리의 contribution을 추가해서, 전반적인 SWD를 향상시키고 이미지의 질을 높인다.

마지막으로, **(h)**에서 제대로 된 네트워크(PGGAN)를 사용하고 더 오랫동안 training 한다 - 결과가 가장 좋다.

<br>

### 6.2. Convergence and Training Speed

<p align="center">
    <img src="..\assets\pggan\figure4.PNG" style="zoom: 80%;" />
    Figure 4
</p>

Figure 4는 SWD metric과 raw image throughput의 관점에서 progressive growing의 효과를 나타낸다. 

NVIDIA Tesla P100을 사용한 single-GPU setup에서 측정했다.

**(a)** 128 x 128 해상도를 가진 CelebA를 Gulrajani et al. (2017) 네트워크에 넣고 training을 한 것이다. 각 그래프는 Laplacian pyramid의 한 레벨에서 sliced Wasserstein distance를 나타내고, 수직선은 Table 1에서 training을 중단한 지점이다.

**(b)** Gulrajani et al. (2017)에 progressive growing을 추가한 그래프이다. 점선으로 된 수직선은 G와 D의 해상도를 두 배로 만들 지점을 나타낸다.

**(C)** 1024 x 1024 해상도에서 raw training speed에서 progressive growing의 효과를 나타낸다.

<br>

**(b)**는 better optimum에 상당히 잘 수렴하고, 총 훈련시간을 약 2배 단축한다. 

Progressive growing이 없으면, generator와 discriminator의 모든 레이어들은 large-scale variation과 small-scale detail을 위해 간결한 intermediate representations를 동시에 찾는다. 그러나 progressive growing이 있으면, 기존의 low-resolution layers는 이미 일찍 수렴될 가능성이 있어서 네트워크는 새로운 레이어가 도입됨에 따라 점점 더 작은 스케일의 효과로 representations를 구체화하는 작업만 담당한다.

실제로, Figure 4**(b)**에서 largest-scale statistical similarity curve (16)이 optimal value에 매우 빠르게 도달한다. Smaller-scale curves (32, 64, 128)는 동등하게 일관성 있게 수렴한다. 

Non-progressive training인 Figure 4**(a)**에서 SWD metric의 각 scale은 예상대로 일정하게 수렴된다.

Figure 4**(c)**에서 progressive growing은 해상도가 증가하면 속도가 증가한다. 

<br>

### 6.3. High-Resolution Image Generator using Celeba-HQ Dataset

High output resolutions에서 결과를 의미있게 설명하기 위해서, 충분히 다양한 고품질 데이터가 필요하다. 그러나, 이전에 GAN 문헌에서 사용된 거의 모든 공개 데이터 세트는 $32^2$에서 $480^2$까지 상대적으로 low resolutions이다. 이를 위해, 1024 x 1024 해상도의 30000개의 이미지들로 구성된 CelebA의 고품질 버전을 만들었다. 

<br>

<p align="center">
    <img src="..\assets\pggan\figure5.PNG" style="zoom:80%;" />
    Figure 5
</p>

CelebA-HQ 데이터셋을 사용해서 만든 1024 x 1024 이미지이다.

<br>

8개의 Tesla V100 GPUs를 가지고 4일동안 train했다. 우리의 implementation은 해당 output resolution에 따른 adaptive minibatch size를 사용해서 유효한 memory budget가 최적으로 사용되었다.

우리의 contributions이 loss function에 독립적이라는 것을 설명하기 위해, WGAN-GP 대신 LSGAN loss를 사용해서 같은 네트워크를 train했다. 

Figure 1은 LSGAN을 사용하여 만든 이미지이다. 

<br>

### 6.4. LUSN Results

<p align="center">
    <img src="..\assets\pggan\figure6.PNG" style="zoom:90%;" />
    Figure 6
</p>

<p align="center">
    <img src="..\assets\pggan\figure7.PNG" style="zoom:80%;" />
    Figure 7
</p>

Figure 6은 LSUN BEDROOM에서 결과를 비교한 것이다.

Figure 7은 $256^2$해상도의 다양한 LSUN categories에서의 결과를 보여준다. 

<br>

### 6.5. CIFAR10 Inception Scores

우리의 방법을 사용해서 8.80이라는 높은 점수를 얻었다.

<br>

<br>

PGGAN의 더 세부적인 정보를 알고 싶으면 논문의 부록을 참고하면 된다.

------

Reference

[https://towardsdatascience.com/progan-how-nvidia-generated-images-of-unprecedented-quality-51c98ec2cbd2](https://towardsdatascience.com/progan-how-nvidia-generated-images-of-unprecedented-quality-51c98ec2cbd2)

[https://wiserloner.tistory.com/1196](https://wiserloner.tistory.com/1196)

[https://sensibilityit.tistory.com/508?category=731657](https://sensibilityit.tistory.com/508?category=731657)

[https://aigong.tistory.com/65](https://aigong.tistory.com/65)