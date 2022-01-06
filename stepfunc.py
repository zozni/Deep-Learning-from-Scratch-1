import numpy as np

def step_function(x):  # 인수 x는 실수만 받아들인다.
    if x > 0:
        return 1
    else:
        return 0

# 넘파이 배열도 지원하도록 수정한 것
def step_function(x):
    y = x > 0
    return y.astype(np.int)  # 넘파이 배열의 자료형을 변환할 때는 astype() 메소드를 이용한다.
                             # 원하는 자료형인 int를 인수로 지정하면 된다.