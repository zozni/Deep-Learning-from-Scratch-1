import numpy as np

def softmax_2(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y
""" 소프트맥스 함수 구현시 주의점 """
# 위에서 구현한 softmax 함수의 코드는 식을 제데로 표현하고 있지만 컴퓨터로 계산할 때는 결함이 있다.
# -> 오버플로우 문제 발생
# 지수 함수가 식에 포함되어 있기 때문에 아주 큰 값을 내밷음. 이런 큰 값끼리 나눗셈을 하면 수치가 불안정해진다.
# 따라서 개선한 수식을 이용한다. 93p

"""위 문제점을 개선한 소프트맥스 함수"""
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)  # 오버플로우 대책
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y, np.sum(y))  # 소프트맥스 함수의 출력의 총합은 1