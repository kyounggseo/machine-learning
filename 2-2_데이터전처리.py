#훈련 세트와 테스트 세트
#길이 특성
fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8, 
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
# 무게 특성
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7, 
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]
fish_data = [[l,w] for l, w in zip(fish_length, fish_weight)]
fish_target = [1]*35 + [0]*14   # 35마리의 도미와 14마리의 방어 구분 
# 사이킷런의 KNeighborsClassifier 클래스를 임포트하고 모델 객체 만들기
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
# fish_data의 다섯 번째 샘플을 출력하기
print(fish_data[4]) # 배열의 인덱스는 0부터 시작 즉 다섯 번째 샘플의 인덱스는 4
# 파이썬 슬라이싱 
# 슬라이싱을 사용할 때는 마지막 인덱스의 원소는 포함하지 않는다.
print(fish_data[0:5]) # 0~4가지의 5개 원소만 선택되고 인덱스 5인 여섯 번째 원소는 선택X
print(fish_data[:5]) # '0:5'와 같이 처음부터 시작되는 슬라이싱의 경우 0 생략 가능
print(fish_data[44:])
# 훈련 세트로 입력값 중 0부터 34번째 인덱스까지 사용
train_input = fish_data[:35]
# 훈련 세트로 타킷값 중 0부터 34번째 인덱스까지 사용
train_target  = fish_target[:35]
# 테스트 세트로 입력값 중 35번째부터 마지막 인덱스까지 사용
test_input = fish_data[35:]
# 테스트 세트로 타킷값 중 35번째부터 마지막 인덱스까지 사용
test_target = fish_target[35:]
kn = kn.fit(train_input, train_target)
print(kn.score(test_input, test_target)) # 정확도가 0이 나옴
# 넘파이
import numpy as np # 파이썬의 대표적인 배열 라이브러리(고차원의 배열을 손쉽게 다룸)
# 파이썬 리스트를 넘파이 배열로 바꾸는 방법은 넘파이 array() 함수에 파이썬 리스트를 전달하면 끝
input_arr = np.array(fish_data)
target_arr = np.array(fish_data)
print(input_arr)
print(input_arr.shape) # 이 명령을 사용하면 (샘플 수,  특성 수)를 출력
np.random.seed(42) # 일정한 결과를 얻기 위해서 초기에 랜덤 시드를 42로 지정
index= np.arange(49) # 넘파이 arange() 함수에 정수 N(= 49)을 전달하면 0에서부터 N-1까지 1씩 증가하는 배열 생성 => 즉, 0부터 48까지 배열 생성
np.random.shuffle(index) # shuffle () 함수는 주어진 배열을 무작위로 섞는 역할 
print(index)
print(input_arr[[1,3]]) #넘파이의 배열 인덱싱 input_arr에서 두 번째와 네 번째 샘플을 선택하여 출력 가능
train_input = input_arr[index[:35]] 
train_target = target_arr[index[:35]]
print(input_arr[13], train_input[0]) #index의 첫 번재 값은 13 즉, train_input의 첫 번째 원소는 input_arr의 열네 번째 원소가 들어있음
test_input = input_arr[index[35:]]
test_target = target_arr[index[35:]]
import matplotlib.pyplot as plt
plt.scatter(train_input[:,0], train_input[:,1]) # 2차원 배열은 행과 열 인덱스를 콤마로 나누어 지정 
plt.scatter(test_input[:,0], test_input[:,1])
plt.xlabel('length')
plt.ylabel('weight')
plt.show # 파란색이 훈련세트, 주활색이 테스트 세트 
# 두 번째 머신러닝 프로그램
# k-최근접 이웃 모델을 훈련
kn = kn.fit(train_input, train_target)
kn.score(test_input, test_target) # 인덱스를 섞어 만든 train_input과 train_target으로 모델을 훈련
# 테스트 세트의 예측 결과
kn.predict(test_input) # predict()메서드는 사이킷런 모델을 훈련하고 예측할 때 사용하는 메서드, 특성 데이터 하나만 매개변수로 받는다.
# 실제 타깃 결과
test_target
# => 정확도 100%