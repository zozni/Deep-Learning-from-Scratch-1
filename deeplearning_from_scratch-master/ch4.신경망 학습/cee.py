""" cross entropy error """

import numpy as np

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta)) # 아주 작은 값인 delta를 더해서 절대 0이 되지 않도록 함.
                                          # 즉, 마이너스 무한대가 발생하지 않도록 함.