"""
 미니배치와 같은 배치 데이터를 지원하는 교차 엔트로피 오차
"""

import numpy as np

def cross_entropy_error(y,t):  # y는 신경망의 출력, t는 정답 레이블
    if y.ndim == 1:                 # 즉 데이터 하나당 교차 엔트로피 오차를 구하는 경우는
        t = t.reshape(1, t.size)    # reshape 함수로 데이터의 형상을 바꿔준다. 
        y = y.reshape(1, y.size)    # 그리고 배치의 크기로 나눠 정규화하고 이미지 1장당 평균의 
                                    # 교차 엔트로피 오차를 계산한다.
    batch_size = y.shape[0]
    return -np.sum(t*np.log(y + 1e-7)) / batch_size

"""
 정답 레이블이 원핫인코딩이 아니라 2나 7 등의 숫자 레이블로 주어졌을 때의 교차 엔트로피 오차
"""

def cross_entropy_error2(y,t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)    # reshape 함수로 데이터의 형상을 바꿔준다. 
        y = y.reshape(1, y.size)

    batch_size1 = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size1), t] + 1e-7)) / batch_size1
# 원핫인코딩일 때 t가 0인 원소는 cee도 0이므로 그 계산은 무시해도 좋다.
# 즉, 정답에 해당하는 신경망의 출력만으로 교차엔트로피 오차를 계산할 수 있다.