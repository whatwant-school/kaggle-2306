# kaggle-2306
ICR - Identifying Age-Related Conditions


## Competition
- https://www.kaggle.com/competitions/icr-identify-age-related-conditions

---

## 01 - 2023-06-24
- Code 부분을 통해 다른 사람들의 내용을 참고할 수 있음
  - 정렬 기준을 통해 원하는 분류 방식으로 찾아보면 도움이 될 것임
- 익명의 column 이다보니, 정확히 의학적으로 어떤 내역인지 파악할 수는 없음
  - 그래서 공공데이터 등을 활용하는 등의 작업은 어려울 것으로 보임
- 평가 지표는 `log loss`로 뽑고 있음
- 주어진 문제는 binary classification problem

### Homework
- Most Votes Top 5 살펴보고 오기
  - 다음 주에 코드 리뷰 방식으로 진행 예정

### Link
- [데이터 설명과 평가지표에 대하여 간략한 정리](https://github.com/sgr1118/ICR_kaggle_Challenge/tree/main)
- [House Price Prediction 샘플 코드](https://github.com/sgr1118/EX/blob/main/%5BExp_03%5D_Kaggle_Challenge_(2019_kaggle_korea).ipynb)

---

## 02 - 2023-07-01

### Kaggle 데이터 불러오기
- Kaggle에서 Token을 새로 생성하면 기존의 token은 무용지물이 된다.

```python
!pip install kaggle

# https://www.kaggle.com/settings - API - Create New Token

from google.colab import files
files.upload()

!ls -lha kaggle.json

!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle competitions download -c icr-identify-age-related-conditions
!ls -al

!unzip icr-identify-age-related-conditions.zip
!ls -al
```

### [Identifying Age-Related Conditions w/ TFDF](https://www.kaggle.com/code/gusthema/identifying-age-related-conditions-w-tfdf)
- [Dicision Forest](https://www.tensorflow.org/decision_forests?hl=ko)를 중심으로 문제를 해결
  - 정형 데이터 분류 시 우수한 성능을 보임

- [Notebook Review]


---

## (SELF-STUDY) Decision Tree & Random Forests


---

## [(SELF-STUDY) Over-Sampling or Under-Sampling (불균형 데이터)](unbalanced-data.md)
