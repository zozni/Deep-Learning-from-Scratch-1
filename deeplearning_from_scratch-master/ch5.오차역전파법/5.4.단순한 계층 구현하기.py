# 5.4.1 곱셈 계층
class MulLayer:
    def __init__(self):  # 인스턴스 변수인 x와 y를 초기화한다.
        self.x = None    # 이 두 변수는 순전파 시의 입력값을 유지하기 위해 사용한다.
        self.y = None

    def forward(self, x, y):   # x와 y를 인수로 받고 두 값을 곱해서 반환한다.
        self.x = x 
        self.y = y
        out = x * y
        return out

    def backward(self, dout):   # 상류에서 넘어온 미분에 순전파때의 값을 서로 바꿔 곱한 후 하류로 흘린다.
        dx = dout * self.y  # x와 y를 바꾼다. 
        dy = dout * self.x
        return dx, dy


# 5.4.2 덧셈 계층
class AddLayer:
    def __init__(self):  # 초기화가 필요없으니 아무일도 하지 않는다.
        pass

    def forward(self, x, y):  # 입력받은 두 인수 x, y를 더해서 반환한다.
        out = x + y
        return out

    def backward(self, dout):  # 상류에서 내려온 미분을 그대로 하류로 흘린다.
        dx = dout * 1
        dy = dout * 1
        return dx, dy

""" buy_apple.py """

if __name__ == '__main__':
    # 문제1의 예시
    apple = 100
    apple_num = 2
    tax = 1.1

    # 계층들
    mul_apple_layer = MulLayer()
    mul_tax_layer = MulLayer()

    # 순전파
    apple_price = mul_apple_layer.forward(apple, apple_num)
    price = mul_tax_layer.forward(apple_price, tax)

    print(price)  # 220.0

    # 역전파          # 각 변수에 대한 미분은 backward()에서 구할 수 있다.
    dprice = 1
    dapple_price, dtax = mul_tax_layer.backward(dprice)
    dapple, dapple_num = mul_apple_layer.backward(dapple_price)

    print(dapple, dapple_num, dtax)  # 2.2 110.0 200
    
    """ backward() 호출 순서는 forward() 때와 반대이다. """
    """ 또 backward()가 받는 인수는 순전파의 출력에 대한 미분이다."""

    # 문제2의 예시
    orange = 150
    orange_num = 3

    # 계층들
    mul_apple_layer = MulLayer()
    mul_orange_layer = MulLayer()
    add_apple_orange_layer = AddLayer()
    mul_tax_layer = MulLayer()

    # 순전파
    apple_price = mul_apple_layer.forward(apple, apple_num)
    orange_price = mul_orange_layer.forward(orange, orange_num)
    all_price = add_apple_orange_layer.forward(apple_price, orange_price)
    price = mul_tax_layer.forward(all_price, tax)

    print(price)  # 715.0

    # 역전파
    dprice = 1
    dall_price, dtax = mul_tax_layer.backward(dprice)
    dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
    dornage, dorange_num = mul_orange_layer.backward(dorange_price)
    dapple, dapple_num = mul_apple_layer.backward(dapple_price)

    print(dapple_num, dapple, dornage, dorange_num, dtax)
    # 110.0 2.2 3.3 165.0 650

""" 필요한 계층을 만들어 forward()를 적절한 순서로 호출한다. 
    그런 다음 순전파와 반대 순서로 역전파 메소드인 backward()를 호출하면 원하는 미분이 나온다. """