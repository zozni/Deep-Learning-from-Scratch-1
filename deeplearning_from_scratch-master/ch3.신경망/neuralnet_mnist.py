import sys
import os
import pickle
import numpy as np
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax

""" mnist 데이터셋을 가지고 추론을 수행하는 신경망을 구현 """

def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():  # pickle파일인 sample_weight.pkl에 저장된 학습된 가중치 매개변수를 읽는다.
                     # 이 파일에는 가중치와 편향 매개변수가 딕셔너리 변수로 저장되어 있다.
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x,W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

""" 위 세 함수를 이용해 신경망에 의한 추론을 수행해보고 정확도를 평가."""

x, t = get_data()
network = init_network()

