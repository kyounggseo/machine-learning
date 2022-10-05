# 도미 데이터 준비하기
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0] #  도미의 길이

bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0] # 도미의 무게

import matplotlib.pyplot as plt #matplotlib의 pyplot 함수를 plt로 줄여서 사용

plt.scatter(bream_length, bream_weight)
plt.xlabel('bream_length') #x축은 길이
plt.ylabel('bream_weight)') #y축은 무게 
plt.show() # console에 plt를 나타냄

# 주황색 점 : 빙어의 산점도 14개
# 방어 데이터 준비하기
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0] # 방어의 길이
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9] # 방어의 무게

plt.scatter(bream_length, bream_weight) # scatter()는 산점도를 그리는 맷플롯립 함수
plt.scatter(smelt_length, smelt_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show() # console에 plt를 나타냄

# 첫 번재 머신러닝 프로그램
length = bream_length + smelt_length
weight = bream_weight + smelt_weight

fish_data = [[l, w] for l, w in zip(length, weight)]

print(fish_data)

fish_target = [1] * 35 + [0] * 14 # [1]은 도미, [0]은 방어를 의미한다. 

print(fish_target)

from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier() # k-최근접 이웃 분류 모델을 만드는 사이킷런 클래스

# 사이킷런 모델을 훈련할 때 사용하는 메서드 - fit()
# 머신러닝 알고리즘이 데이터에서 규칙을 찾는 과정을 훈련이라 하고 사이킷런에서는 fit() 메서드가 담당 
kn.fit(fish_data, fish_target) 

result = kn.score(fish_data, fish_target)
print('kn.score =', result * 100, '%')

# k-최근접 이웃 알고리즘
plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.scatter(30, 600, marker='^') # 도미에 가까운 것을 알 수 있다.
# 만약 plt.scatter(20, 170, marker='^')으로 변경할 시 출력 값은 도미로 나오지만 그래프 상에서는 애매한 위치를 갖게 된다 
# => 분석하고자 하는 데이터가 많아야 정확한 값을 알 수 있다는 점을 알 수 있다.
plt.xlabel('length')
plt.ylabel('weight)')
plt.show() # console에 plt를 나타냄

test = kn.predict([[30, 600]]) # 사이킷런 모델을 훈련하고 예측할 때 사용하는 메서드- predict()
print('kn.predict =' , test)

print(kn._fit_X)

print(kn._y)

kn49 = KNeighborsClassifier(n_neighbors=49) #참고 데이터를 49개로 한 kn49 모델

kn49.fit(fish_data, fish_target)
kn49.score(fish_data, fish_target) # 훈련된 사이킷런 모델의 성능 측정

print(35/49)