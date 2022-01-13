# coding: utf-8
import sys, os
import numpy as np
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from common.functions import *
from common.gradient import numerical_gradient


class TwoLayerNet:  # 2층 신경망을 하나의 클래스로 구현

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화        params 변수에 저장된 가중치 매개변수가 예측처리(순방향 처리)에서 사용된다.
        self.params = {} # 신경망의 매개변수를 보관하는 딕셔너리 변수 (인스턴스 변수)
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):   # 예측을 수행
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
    
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y
        
    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):   # 손실함수의 값을 구함
        y = self.predict(x)  # predict()의 결과와 정답 레이블을 바탕으로 cee를 구하도록 구현.
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):   # 정확도를 구함
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x : 입력 데이터, t : 정답 레이블
    def numerical_gradient(self, x, t):   # 각 가중치 매개변수의 기울기를 구함
        loss_W = lambda W: self.loss(x, t)  # 이 메소드를 사용해 기울기를 계산하면 grads 변수에 기울기 정보가 저장된다.
        # 수치미분 방식으로 각 매개변수의 손실함수에 대한 기울기를 계산한다.
        grads = {}  # 기울기 보관하는 딕셔너리 변수(numerical_gradient()메소드의 반환 값)
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    def gradient(self, x, t):   # 가중치 매개변수의 기울기를 구함. (numerical_gradient()의 성능 개선판)
        W1, W2 = self.params['W1'], self.params['W2']  # 오차역전파 법을 사용하여 기울기를 효과적으로 계산 (속도good)
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y - t) / batch_num  # ????
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads
