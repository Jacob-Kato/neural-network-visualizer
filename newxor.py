import random
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Neuron:
    def __init__(self, numw):
        self.w = [random.uniform(-0.5,0.5) for w in range(numw)]
        self.bias = random.uniform(-0.5,0.5)
        self.lastz = None
        self.last_act = None

    def forward(self,x, act):
        print(f"w:{self.w},x:{x}")
        self.lastz = sum(x*w for x,w in zip(x,self.w))+self.bias
        if act:
            self.last_act = self.sigmoid(self.lastz)
            return self.last_act
        else:
            return self.lastz

    def sigmoid(self, x):
        return 1/(1+math.exp(-x))

    def sigmoid_derivative(self,x):
        sig = 1/(1+math.exp(-x))
        return sig * (1- sig)

class Network:
    def __init__(self):
        self.output = Neuron(1)

    def forward(self,x):
        y = self.output.forward(x,False)
        return y
    def backwards(self,x,target,lr=0.5):
        guess = self.output.lastz
        error = guess - target
        del3 = error * 1
        for i in range(len(self.output.w)):
            grad = del3 * x[i]
            self.output.w[i] -= lr * grad

        self.output.bias -= lr * del3


network = Network()

dataset = [[[0.1],0.2],
           [[0.2],0.4],
           [[0.4],0.8],]

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,4))
epoch_data, loss_data = [],[]

def animate(epoch):
    total_loss = 0

    for x,y in dataset:
        guess = network.forward(x)
        total_loss += (guess - y)**2 
        network.backwards(x,y)

    
    epoch_data.append(epoch)
    loss_data.append(total_loss)


    ax1.clear()
    ax1.plot(epoch_data,loss_data, color='red')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss map")

    ax2.clear()
    truex = [d[0][0]for d in dataset]
    truey = [d[1] for d in dataset ]
    ax2.scatter(truex, truey, color='green', label='True data', zorder=5)

    linex = [0.0, 0.5]
    liney = [network.forward([x]) for x in linex]
    ax2.plot(linex, liney, color='red', label="Network pre")

    ax2.set_xlim(0,0.5)
    ax2.set_ylim(0,1.0)
    ax2.set_xlabel("Input (x)")
    ax2.set_ylabel("Output (Y)")
    ax2.set_title(f"Epoch {epoch}")
    ax2.legend()


    return ax1,ax2


anti = animation.FuncAnimation(fig,animate,frames=200,interval=50,repeat=False)
plt.tight_layout()
plt.show()
