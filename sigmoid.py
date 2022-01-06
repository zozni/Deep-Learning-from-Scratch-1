import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))   # 인수 x가 넘파이 배열이어도 올바른 결과가 나온다.

x = np.array([-1.0, 1.0, 2.0])
print(sigmoid(x))

# 넘파이의 브로드캐스트 기능으로 인해 이 함수는 넘파이 배열도 처리가능.

# 브로드캐스트 기능이란 
# 넘파이 배열과 스칼라값의 연산을 
# 넘파이 배열의 원소 각각과 스칼라값의 연산으로 바꿔 수행하는 것이다.

# 시그모이드를 그래프로
x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1) # y축 범위 지정
plt.show()