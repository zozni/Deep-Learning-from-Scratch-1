import sys
import os
import numpy as np
sys.path.append(os.pardir)
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


# 4.4.2 신경망에서의 기울기
class simpleNet:   # 형상이 2x3인 가중치 매개변수 하나를 인스턴스 변수로 갖는다 .
    """docstring for simpleNet"""
    def __init__(self):
        self.W = np.random.randn(2, 3)  # 정규분포로 초기화

    def predict(self, x):            # 예측을 수행
        return np.dot(x, self.W)

    def loss(self, x, t):           # 손실함수의 값을 구함 
        z = self.predict(x)         # x는 입력데이터, t는 정답레이블
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


net = simpleNet()
print(net.W)  # 가중치 매개변수(랜덤)
x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)
print(np.argmax(p))  # 최댓값의 인덱스

t = np.array([0, 0, 1])  # 정답 레이블
print(net.loss(x, t))


def f(W):
    return net.loss(x, t)
# 새로운 함수를 정의하는데 위 문법처럼 하는 거말고도 
# f = lambda w: net.loss(x,t) 이렇게 람다 기법을 쓸 수도 있음! @@@@@@@@@@

dW = numerical_gradient(f, net.W)  # 기울기를 구함.
print(dW)
