import numpy as np

def relu(x):
    return np.maximum(0,x)   # maximum은 두 입력 중 큰 값을 선택해 반환하는 함수


A = np.array([1,2,3,4])
print(A)
i = np.ndim(A)  # 배열의 차원 수
print(i)

j = A.shape  # 배열의 형상. A는 1차원 배열이고 원소 4개로 구성.
print(j)

k = A.shape[0]
print(k)