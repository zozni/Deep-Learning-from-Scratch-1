import numpy as np

A = np.array([[1, 2], [3, 4]])
print(A.shape)

B = np.array([[5, 6], [7, 8]])
print(B.shape)

print(np.dot(A,B)) # 두 행렬의 곱을 계산 
                   # 입력이 1차원배열이면 벡터를 계산하고
                   # 입력이 2차원배열이면 행렬 곱을 계산한다.

                   # 다차원 배열을 곱하려면 두 행렬의 대응하는 차원의 원소 수를 일치시켜야 한다.