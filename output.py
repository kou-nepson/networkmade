import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.random.randn(10, 2)
W1 = np.random.randn(2, 4)
b1 = np.random.randn(4)
W2 = np.random.randn(4, 3)
b2 = np.random.randn(3)

print("入力の値=\n" , x)
h = np.dot(x, W1) + b1
print("中間層ニューロンの値\n" , h)
a = sigmoid(h)
print("出力層のニューロンの値=\n" , a)
s = np.dot(a, W2) + b2
print(s)
