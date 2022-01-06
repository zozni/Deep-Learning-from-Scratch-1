import numpy as np
import matplotlib.pylab as plt

def step_function(x):                  # 인수로 받은 넘파이 배열의 원소 각각을 인수로 계단 함수 실행해,
    return np.array(x > 0, dtype=int)    # 그 결과를 다시 배열로 만들어 돌려준다.

x = np.arange(-5.0, 5.0, 0.1)  # -5.0에서 5.0 전까지 0.1 간격의 넘파이 배열을 생성한다.
y = step_function(x)
plt.plot(x,y)
plt.ylim(-0.1, 1.1)  # y축의 범위 지정
plt.show()