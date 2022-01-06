import numpy as np
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from PIL import Image

# 3.6.1 MNIST 이미지 확인해보기


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, # flatten=True로 설정해 읽어들인 이미지는 
    normalize=False)                                           # 1차원 넘파이 배열로 저장되어 있다.
                                            #그래서 이미지를 표시할 때는 원래 형상인 28x28 크기로 다시 변형해야 한다.
img = x_train[0]
label = t_train[0]
print(label)  # 5
print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 원래 이미지 모양으로 변형
print(img.shape)  # (28, 28)

img_show(img)
