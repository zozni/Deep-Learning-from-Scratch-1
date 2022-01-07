import sys
import os
import pickle
import numpy as np
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax

def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test 

def init_network(): 
    with open("sample_weight.pkl", 'rb', "utf-8") as f:
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

x, t = get_data()
network = init_network()

batch_size = 100 # 배치 크기
accuracy_cnt = 0
                                            # range(start, end)
for i in range(0, len(x), batch_size):   #range 함수는 인수를 2개 지정해 호출하면 start에서 end-1까지의 정수로 이루어진
    x_batch = x[i:i+batch_size]          # 리스트를 반환한다.
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)  
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

"""axis=1은 100x10의 배열 중 1번째 차원을 구성하는 각 원소에서 최댓값의 인덱스를 찾도록 한 것"""

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

# range(start, end, step) 처럼 인수를 3개 지정하면 start에서 end-1 까지 step 간격으로 증가하는 리스트를 반환한다.
