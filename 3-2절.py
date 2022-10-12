import numpy as np

perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

# 사이키런 train_test.spli() 함수를 사용해 훈련 세트와 테스트세트로 나누기 
from sklearn.model_selection import train_test_split

# 훈련 세트와 테스트 세트로 나눈다.
train_input, test_input, train_target , test_target = train_test_split(
        perch_length, perch_weight, random_state=42) # 책과 동일한 결과가 나오기 위해서 randem_state=42로 지정하기

# 훈련 세트와 테스트 세트를 2차원 배열로 나눈다.
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)

# ## 결정 계수(R^2)
# 최근접 이웃 계수를 3으로 하는 모델을 훈련
from sklearn.neighbors import KNeighborsRegressor

knr = KNeighborsRegressor(n_neighbors=3)

# k-최근접 이웃 회귀 모델을 훈련
knr.fit(train_input, train_target)

# 길이가 50cm인 농어의 무게 예측
print(knr.predict([[50]]))

# 위의 예측 값과 실제 농어 무게 값의 차이가 있음 => 훈려 세트와 50cm농어, 이 농어의 최근접 이웃이 어떤 형태를 띄고 있는지 산점도로 표시
import matplotlib.pyplot as plt

# 50cm 농어의 이웃을 구하기
distances, indexes = knr.kneighbors([[50]])

# 훈련 세트의 산점도 그리기
plt.scatter(train_input, train_target) # scatter()는 산점도를 그리는 함수 

plt.scatter(train_input[indexes], train_target[indexes], marker = 'D')

# 50cm 농어 데이터
plt.scatter(50, 1033, marker = '^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
# 길이가 커질수록 농어의 무게가 증가하는 경향

# 이웃 샘플의 타깃의 평균
print(np.mean(train_target[indexes]))
# k-최근접 이웃 회귀는 가장 가까운 샘플을 찾아 타깃을 평균한다.
# 즉, 새로운 샘플이 훈련 세트의 범위에 벗어나면 엉둥한 갑을 예측 할 수 있다.

# 예를 들어, 길이가 100cm인 농어도 길이가 50cm인 농어의 무게와 같다.
print(knr.predict([[100]]))

# 100cm 농어의 이웃을 구하기
distances, indexes = knr.kneighbors([[100]])

# 훈련 세트의 산점도 그리기
plt.scatter(train_input, train_target) # scatter()는 산점도를 그리는 함수 

plt.scatter(train_input[indexes], train_target[indexes], marker = 'D')

# 100cm 농어 데이터
plt.scatter(100, 1033, marker = '^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
# 농어가 아무리 커도 무게가 늘어나지 않는다.

## 선형회귀를 이용해서 해결 가능
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

# 선형 회귀 모델을 훈련
lr.fit(train_input, train_target)

# 50cm 농어에 대해 예측
print(lr.predict([[50]]))
# k-최근접 이웃 회귀를 사용했을 때보다 선형 회귀는 50cm 농어의 무게를 아주 높게 예측했다. 

print(lr.coef_, lr.intercept_)

# 훈련 세트의 산점도
plt.scatter(train_input, train_target) # scatter()는 산점도를 그리는 함수 

# 15에서 50까지 1차 방정식 그래프 그리기
plt.plot([15, 50], [15*lr.coef_+lr.intercept_, 50*lr.coef_+ lr.intercept_])

# 50cm 농어 데이터
plt.scatter(50, 1241.8, marker = '^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
# 여기서 나온 직선이 선형 회귀 알고리즘이 이 데이터 세에서 찾은 최적의 직선

# 훈련 세트 범위를 버어난 농어의 무게도 예측
print(lr.score(train_input, train_target)) # 훈련 세트
print(lr.score(test_input, test_target)) # 테스트 세트

## 다향 회귀
# 그래프 왼쪽 아래가 이상함. 농어는 0g이 될 수 없음 => 즉, 최적의 직선이 아닌 최적의 곡선을 찾기 (2차 방정식의 그래프 그리기)

# 앞에서 배운 column_stack() 함수를 사용하여 2차 방정식 구현 가능
train_poly = np.column_stack((train_input ** 2, train_input))
test_poly = np.column_stack((test_input ** 2, test_input))

# train_input ** 2 식에도 넘파이 브로드캐스팅이 적용됨. 즉, train_input에 있는 모든 원소를 제곱.
print(train_poly.shape, test_poly.shape)

# train_poly를 사용해 선형 회귀 모델을 다시 훈련
lr = LinearRegression()
lr.fit(train_poly, train_target)

# 앞에서 예측한 값보다 더 높은 값을 예측
print(lr.predict([[50**2, 50]]))

# 훈련한 계수와 절편 출력
print(lr.coef_, lr.intercept_)
# 이 모델은 ""무게 = 1.01 * 길이^2 - 21.6 * 길이 + 116.05""와 같은 그래프를 학습한 것.

# 2차 방정시의 계수와 절편 a,b,c를 알았으니 훈련세트의 산점도 그리기
# 구간별 직선을 그리기 위해 15에서 49가지 정수 배열 만들기
point = np.arange(15,50)

# 훈련 세트의 산점도 그리기
plt.scatter(train_input, train_target)

# 15에서 49까지 2차 방정식 그래프 그리기
plt.plot(point, 1.01*point**2 - 21.6*point + 116.05)

# 50cm 농어 데이터
plt.scatter(50,1547, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
# 앞에서 그린 단순 선형 회귀 그래프보다 나은 그래프가 그려졌다. 

# 훈련 세트와 테스트 세트의 R^2 점수를 평가
print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))