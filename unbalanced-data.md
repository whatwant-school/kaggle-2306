# 불균형 데이터
- 클래스 別 관측치의 수가 현저하게 차이가 나는 데이터
  - 암 발생률과 같은 경우 발병하는 경우가 극소수 → 불균형
- Study Reference
  - [불균형 데이터 분석을 위한 샘플링 기법](https://www.youtube.com/watch?v=Vhwz228VrIk)


## why problem?
- 일반적으로 이상(소수)을 정확히 분류하는 것이 중요
- 관측된 이상 데이터값이 실제 데이터를 충분히 반영하지 못함
  - 실제 이상 값임에도 정상으로 잘못 판단할 수 있음


## 성능 평가 (Accuracy)
- Confusion matrix (정오행렬)

<table>
<thead>
  <tr>
    <th colspan="2" rowspan="2"></th>
    <th colspan="2">예측</th>
  </tr>
  <tr>
    <th>이상</th>
    <th>정상</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="2">실제</td>
    <td>이상</td>
    <td>A</td>
    <td>B</td>
  </tr>
  <tr>
    <td>정상</td>
    <td>C</td>
    <td>D</td>
  </tr>
</tbody>
</table>

- 예측 정확도
  - $Accuracy (정확도) = {A + D \over A + B + C + D}$


## Example
- 하지만, 아래와 같은 경우 올바른 것인가?
  - 이상이 있음에도 정상으로 잘못 판정한 것이 5나 되는데...

<table>
<thead>
  <tr>
    <th colspan="2" rowspan="2"></th>
    <th colspan="2">예측</th>
  </tr>
  <tr>
    <th>이상</th>
    <th>정상</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="2">실제</td>
    <td>이상</td>
    <td>5</td>
    <td>5</td>
  </tr>
  <tr>
    <td>정상</td>
    <td>0</td>
    <td>40</td>
  </tr>
</tbody>
</table>

- 예측 정확도
  - $Accuracy (정확도) = {5 + 40 \over 5 + 5 + 0 + 40} = {45 \over 50} = 0.9$

- 높은 예측 정확도를 보이지만, 모델 성능에 대한 왜곡


## 불균형 데이터 해결 방안 2가지
- 데이터를 조정해서 해결
  - Sampling method (샘플링 기법)
- 모델을 조정해서 해결
  - Const sensitive learning (비용 기반 학습)
  - Novelty detection (단일 클래스 분류 기법)

---

# Sampling method (샘플링 기법)
- Undersampling (다수 데이터를 줄여서 소수 데이터 만큼)
  1. Random undersampling
  2. Tomek links
  3. Condensed Nearest Neighbor Rule
  4. One-sided selection
- Oversampling (소수 데이터를 증폭해서 다수 데이터 만큼)
  1. Resampling
  2. SMOTE
  3. Borderline-SMOTE
  4. ADASYN

---

# Undersampling
- 장점
  - 다수 범주 관측치 제거로 계산 시간 감소
  - 데이터 클랜징으로 클래스 오버랩 감소 가능
- 단점
  - 데이터 제거로 인한 정보 손실 발생

## 1. Random undersampling
- 다수 범주에서 무작위로 sampling해서 삭제
- 무작위로 sampling → 할 때마다 다른 결과

## 2. Tomek links
- 두 범주 사이를 탐지하고 정리를 통해 부정확한 분류 경계선 방지

$$Tomek links = d(x_i, x_k) < d (x_i, x_j) \ 또는 \ d(x_j, x_k) < d(x_i, x_j)가 \ 되는 \ 관측치 \ x_k가 \ 없는 \ 경우$$

- 즉, 서로 다른 클래스에 속한 관측치 사이에 다른 관측치 값이 존재하지 않는 link가 형성되는 경우를 의미
- 이렇게 형성된 link에 속한 관측치 중에서 다수 범주에 속한 관측치 삭제

## 3. Condensed Nearest Neighbor Rule (CNN)
- 소수 범주 전체와 다수 범주에서 무작위 하나의 관측치를 선택하여 서브데이터 구성 → 1-NN 알고리즘을 통해 원데이터 분류
  1. 다수 범주의 임의의 관측치 선택
  2. 다수 범주 관측치 하나 선택 → 1-NN 알고리즘에 의해 → 1에서 선택된 관측치와의 거리와 소수 범주 관측치와의 거리 비교 → 분류
- 1-NN에서 k값을 1로 하는 이유
  - 2 이상일 경우 다수 범주에 속하는 기준 데이터가 1개 밖에 없기 때문에 모든 분류가 소수 범주로 선택됨

## 4. One-Side Selection (OSS)
- Tomek links + CNN
  - Tomek links를 이용해서 경계에 있는 다수 범주의 관측치 삭제
  - CNN을 통해서 다수 범주의 멀리 있는(완전히 정상인) 관측치 삭제

---

# Oversampling
- 장점
  - 정보 손실이 없음
  - 대부분의 경우 undersampling에 비해 높은 분류 정확도
- 단점
  - 과적합 가능성
  - 계산 시간이 증가
  - noise 또는 outliar에 민감

## 1. Resampling
- 소수 범주 내 관측치를 복사해서 증폭
  - 의외로 효과적인 경우가 많다
- 단점: 소수 클래스에 과적합 발생할 가능성이 있음

## 2. SMOTE (Synthetic Minority Oversampling Technique)
- [SMOTE: Synthetic Minority Over-sampling Technique](https://arxiv.org/abs/1106.1813)
- 소수 범주에서 가상의 데이터를 생성
- $Synthetic = X + u \cdot (X(nn) - X)$
  - $Synthetic$: 생성된 가상의 관측치
  - nn: nearest neighbors
    - K-NN 처럼, K 값을 필요 → 소수 클래스 특정 관측치의 neighbor K개 선정 → 그 중 random한 1개 선택
  - $X(nn)$: 주변 관측치
  - X: 소수 클래스 관측치
  - $u$: 균등 분포(uniform distribution), $unif(0, 1)$
- example
  - 소수 클래스 관측치 중 하나 선정: $X = (5,1)$
  - 선정된 관측치의 Nearest Neighbor 중 하나 선정: $X(nn) = (2,3)$
  - $(X(nn) - X) = (2,3) - (5,1) = (-3,2)$
  - $Synthetic = X + u \cdot (X(nn) - X) = (5,1) + u(0,1) \cdot (-3,2)$
  - $u(0,1)$의 의미는 0~1 사이의 값 → (5,1) ~ (-3,2) 사이의 임의의 점(관측치)
- 주의 사항
  - K=1 로 하면 안된다

## 3. Borderline-SMOTE
- [Borderline-SMOTE: A New Over-Sampling Method in Imbalanced Data Sets Learning](https://sci2s.ugr.es/keel/keel-dataset/pdfs/2005-Han-LNCS.pdf)
- Borderline 부분만을 sampling
- Howto
  1. Borderline을 찾는다.
      - 소스 클래스 $x_i$에 대해서 k개 주변을 탐색 → k개 중 다수 클래스의 수를 확인
      - 다수 클래스의 수가 적으면 = (0 <= 갯수 <= k/2) = Borderline X = Safe 관측치
      - 다수 클래스의 수가 많으면 = ( k/2 < 갯수 < k) = Borderline O = Danger 관측치
      - 다수 클래스의 수가 k 값이면 = (k = k) = Borderline X = Noise 관측치
  2. Danger 관측치에 대해서만 SMOTE 적용
      - 소수 클래스 $x_i$에 대해서 k개의 소수 클래스를 탐색
      - 각 소수 클래스에 대해 s개 만큼의 샘플을 생성 (s < k)

## 4. ADASYN (ADAptive SYNthetic sampling approach)
- [ADASYN: Adaptive synthetic sampling approach for imbalanced learning](https://ieeexplore.ieee.org/document/4633969)
- sampling하는 개수를 위치에 따라 다르게 하자
- 모든 소수 클래스에 대해 주변을 K개 만큼 탐색하고, 그 중 다수 클래스 관측치의 비율을 계산 → $r_i$ 구하기
- Howto
  1. $r_i = {\mathit{\Delta}_i \over K} \quad i = 1, ... , m$
      - $r_i$: sampling하는 개수
      - $\mathit{\Delta}_i$: 소수 클래스 $x_i$의 주변 K개 中 다수 클래스의 관측치 개수
      - $m$: 소수 클래스 內 관측치 총 개수
  2. $r_i$ 값을 구한 뒤 scaling 하고, G값을 곱한 수 만큼 sampling
      - G = 다수 클래스 개수 - 소수 클래스 개수
  3. 각 소수 클래스를 seed로 하여 할당된 개수만큼 SMOTE 적용


## (최신기법) GAN (Generative Adversarial Nets)
- Howto
  1. 무작위로 noise 생성 $N(0,1)$
  2. generator를 통해 가짜 sample 생성
  3. discriminator에서 진짜/가짜 판별
  4. 반복적인 generator 업데이트를 통해 진짜 sample과 유사하게 data 생성

---

# Const sensitive learning (비용 기반 학습)
- 불균형 데이터 해결 방안 中 모델을 조정해서 해결하는 방법 중 하나
  - Const sensitive learning (비용 기반 학습)
  - Novelty detection (단일 클래스 분류 기법)

## more important
- 이상(소수)을 정상(다수)으로 분류 vs 정상(다수)을 이상(소수)으로 분류
  - 보다 치명적 = 이상(소수)을 정상(다수)으로 분류
- 비용 기반 데이터 가중치 부여
  - 정상:이상 = 1:10

## 비용 기반 데이터 가중치 부여
- Cost-sensitive Decision Tree (의사 결정 나무에 적용)
  - Decision Threshold(결정 임계치)에 반영
  - Split Criteria(분할 기준)에 반영
  - Pruning(가지치기)에 반영
- Cost-sensitive neural network(인공 신경망에 적용)

---

# Novelty detection (단일 클래스 분류 기법)
- 불균형 데이터 해결 방안 中 모델을 조정해서 해결하는 방법 중 하나
  - Const sensitive learning (비용 기반 학습)
  - Novelty detection (단일 클래스 분류 기법)

## Why 두 범주를 모두 고려해야 하나?
- 다수 범주만 고려해서 분류

## Howto
- 가능한한 모든 다수 범주를 포함하는 분류 경계선 생성
  - 중심을 기준으로 다수 범주를 잘 설명하는 boundary(like circle) 생성

## Apply
- Support Vector Data Descripyion (SVDD)