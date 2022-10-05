# 데이터 준비
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

# 위의 데이터가 어떤 형태를 띄고 있는지 산점도로 표시
import matplotlib.pyplot as plt
plt.scatter(perch_length, perch_weight) # scatter()는 산점도를 그리는 함수 
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
# 농어의 길이가 커짐에 따라 무게가 늘어남. 

# 사이키런 train_test.spli() 함수를 사용해 훈련 세트와 테스트세트로 나누기 
from sklearn.model_selection import train_test_split
train_input, test_input, train_target , test_target = train_test_split(
        perch_length, perch_weight, random_state=42) # 책과 동일한 결과가 나오기 위해서 randem_state=42로 지정하기

# 사이킷런에서 사용할 훈련 세트는 2차원 배열이어야 한다. 
test_array = np.array([1,2,3,4])
print(test_array.shape) # test_array 은 (4, )로 출력 1차원 배열을 2차원 배열로 바꾸어야 함.

# 넘파이 배열 중 배열의 크기를 바굴 수 있는  reshape() 메서드 => 바꾸려는 배열의 크기를 지정할 수 있다.
test_array = test_array.reshape(2,2) 
print(test_array.shape) # => (2, 2)로 바뀜

# 첫 번째 크기를 나머지 원소의 개수로 채우고, 두 번쨰 크기를 1로 결정한다는 의미
train_input = train_input.reshape(-1, 1) 
test_input = test_input.reshape(-1, 1)

# 2차원 배열로 변환되었는지 출력
print(train_input.shape, test_input.shape)

# ## 결정 계수(R^2)
# 사이킷런에서 K-최근접 이웃 알고리즘을 구현한 클래스는 KNeighborsRegressor이다. 
# 이는 앞에서 배운 KNeighborsClassifier 과 사용법은 비슷하다.
from sklearn.neighbors import KNeighborsRegressor

knr = KNeighborsRegressor()

# k-최근접 이웃 회귀 모델을 훈련
knr.fit(train_input, train_target)

# 테스트 세트의 점수 확인
print(knr.score(test_input, test_target)) 
#여기서 나온 점수를 결정계수라고 하며, R^2 = 1 - (타깃 - 예측)^2의 합 / (타깃 - 평균)^2의 합으로 계산한다.
# => 사이킷런의 score() 메서드가 출력하는 값이 높을 수록 좋은 것이다. 이 값에서는 예측이 타깃에 아주 가까워지기에 1에 가까운 값이 출력된 것이다. 

# mean_absolute_error를 사용해 타깃과 예측의 절댓값 오차를 평균하여 반환하기
from sklearn.metrics import mean_absolute_error

# 테스트 세트에 대한 예측을 만든다.
test_prediction = knr.predict(test_input)

# 테스트 세트에 대한 평균 절댓값 오차를 게산한다. 
mae = mean_absolute_error(test_target, test_prediction)
print(mae)
# 결과에서 예측이 평균적으로 19g 정도 타깃값과 다르다는 것을 알 수 있다. 

# ## 과대 적합 vs 과소 적합
# 여기서는 훈련 세트를 사용해 평가해보기로 한다. 
# score() 메서드에 훈련 세트를 전달하여 점수를 출력해 보는 것이다.

#  앞에서 훈련한 모델을 사용해 훈련 세트의 R^2 점수를 확인하기
print(knr.score(train_input, train_target))
# 앞에서 나온 테스트 세트를 사용한 점수와 비교하면 훈련 세트보다 테스트 ㅔ트의 점수가 더 높게 나와다 => 과소 적합
# 과대 적합이란? 훈련 세트에서는 점수가 좋게 나왔지만, 테스트 세트에서는 점수가 나쁘게 나왔다면, 이를 훈련 세트에 과대적합 되었다고 말함.
# 과소 적합이란? 과대 적합의 반대 개념 => 원인 : 훈련 세트와 테스트 세트의 크기가 매우 자기 때문에 일어난다.

# 과소 적합 모델을 해결하기 위해서는 모델을 복잡하게 만들면 해결 가능
# 이웃의 개수를 3으로 설정
knr.n_neighbors =3 

# 모델을 다시 훈련
knr.fit(train_input, train_target)
print(knr.score(train_input, train_target)) # k 값을 줄였더니 훈련 세트의 R^2 점수가 높아졌다.

# 테스트 세트의 점수 확인
print(knr.score(test_input, test_target)) # 테스트 세트의 점수는 훈련 세트보다 낮아졌으므로 과소 적합 문제를 해결
