import numpy as np
import matplotlib.pyplot as plt

config = {
    'lr': 0.02, 
    'reg': 5e-4,
    'layerlist': [784, 256, 128, 26],
    'num_epochs': 50,
    'batch_size': 128,
    'seed': 0,
    'data_path': 'E:\Code\emnist-letters-train.csv'
}

eps = 1e-7
lr_decay = 0.95

np.random.seed(config['seed'])

data = np.loadtxt(config['data_path'], delimiter=",")
x = data[:, 1:] / 255.0
y = data[:, 0].astype(int)

idx = np.random.permutation(len(x))
x = x[idx]
y = y[idx]
z = np.zeros((len(y), 26))
for i in range(0, len(y)):
    z[i, y[i] - 1] = 1
ratio = 0.9
split = int(len(x) * ratio)
x_train, x_test = x[0: split], x[split: len(x)]
z_train, z_test = z[0: split], z[split: len(x)]

mean = np.mean(x_train)
std = np.std(x_train)
x_train = (x_train - mean) / (std + eps)
x_test = (x_test - mean) / (std + eps)

class Layer:
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, grad):
        raise NotImplementedError
    
class Linear(Layer):
    def __init__(self, num_in, num_out):
        self.num_in = num_in
        self.num_out = num_out
        self.W = np.random.randn(num_in, num_out) * np.sqrt(2 / num_in)
        self.b = np.zeros((1, num_out))
        
    def forward(self, x):
        self.x = x
        self.y = x @ self.W + self.b
        return self.y
    
    def backward(self, grad):
        self.grad_W = self.x.T @ grad
        self.grad_b = np.sum(grad, axis=0, keepdims=True)
        grad = grad @ self.W.T
        return grad
        
    def update(self, lr):
        self.W -= lr * self.grad_W
        self.b -= lr * self.grad_b
        
class ReLU(Layer):
    def forward(self, x):
        self.x = x
        self.y = np.maximum(x, 0)
        return self.y
    
    def backward(self, grad):
        return grad * (self.x > 0)
    
class SoftmaxCrossEntropy(Layer):
    def forward(self, x, t):
        self.t = t
        x_max = np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x - x_max)
        self.y = exp_x / np.sum(exp_x, axis=1, keepdims=True) # 缓存 self.y 供 backward 使用
        loss = -np.sum(self.t * np.log(self.y + eps)) / self.y.shape[0]
        return loss
    
    def backward(self, grad=None):
        batch_size = len(self.t)
        return (self.y - self.t) / batch_size
    
activation_dict = {
    'relu': ReLU,
}


class MLP:
    def __init__(self, layerlist):
        self.layers = []
        num_in = layerlist[0]
        self.loss_layer = SoftmaxCrossEntropy()
        for num_out in layerlist[1: -1]:
            self.layers.append(Linear(num_in, num_out))
            self.layers.append(activation_dict['relu']())
            num_in = num_out
        self.layers.append(Linear(num_in, layerlist[-1]))
        
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def loss(self, x, t):
        logits = self.forward(x)
        return self.loss_layer.forward(logits, t)
    
    def backward(self):
        grad = self.loss_layer.backward()
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
                
    def evaluate(self, x, t):
        logits = self.forward(x)
        loss = self.loss_layer.forward(logits, t) 
        pred = np.argmax(logits, axis=1)
        label = np.argmax(t, axis=1)
        acc = np.mean(pred == label)
        return loss, acc
        

class SGD:
    def __init__(self, lr, reg):
        self.lr = lr
        self.reg = reg
    
    def update(self, layers):
        for layer in layers:
            if hasattr(layer, 'W'):
                layer.W -= self.lr * (layer.grad_W + self.reg * layer.W)
                layer.b -= self.lr * layer.grad_b
                

mlp = MLP(config['layerlist'])
optimizer = SGD(config['lr'], reg=config['reg'])

losses = []
test_losses = []
test_accs = []
for epoch in range(config['num_epochs']):
    idx = np.random.permutation(len(x_train))
    x_train = x_train[idx]
    z_train = z_train[idx]
    st = 0
    loss = 0.0
    num_epoch = 0
    while True:
        ed = min(st + config['batch_size'], len(x_train))
        if st >= ed:
            break
        x_batch = x_train[st: ed]
        z_batch = z_train[st: ed]
        loss += mlp.loss(x_batch, z_batch)
        mlp.backward()
        optimizer.update(mlp.layers)
        st += config['batch_size']
        num_epoch += 1
    
    losses.append(loss / num_epoch)
    test_loss, test_acc = mlp.evaluate(x_test, z_test)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    optimizer.lr *= lr_decay
    config['lr'] = optimizer.lr
    
print('测试精度：', test_accs[-1])

# 将损失变化进行可视化
plt.figure(figsize=(16, 6))
plt.subplot(121)
plt.plot(losses, color='blue', label='train loss')
plt.plot(test_losses, color='red', ls='--', label='test loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Cross-Entropy Loss')
plt.legend()

plt.subplot(122)
plt.plot(test_accs, color='red')
plt.ylim(top=1.0)
plt.xlabel('Step')
plt.ylabel('Accuracy')
plt.title('Test Accuracy')
plt.show()
