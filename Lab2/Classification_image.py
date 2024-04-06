import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
f=open('output.txt','w')
from PIL import Image
import os

mark='1'
imgs =  []
labels = []
imgpath = []
for root, dirs, files in os.walk("./train_2022"):
    for name in files:
        if name == '.DS_Store': continue
        if root.split('/')[-1]=='.ipynb_checkpoints': continue
        img = Image.open(os.path.join(root, name))
        img = np.asarray(img,dtype='float32')
        imgs.append(img)
        labels.append(int(root.split('/')[-1]))

indices = list(range(len(imgs)))
random.shuffle(indices)
# randimg = np.random.choice(len(imgs),10)
inputs = np.array(imgs)[indices]
targets = np.array(labels)[indices]
# print(f'inputs.shape:{inputs.shape}')
# print(f'targets.shape:{targets.shape}')
dim=inputs[0].shape[0]**2
N=len(imgs)
data0=np.transpose(np.reshape(inputs, [N,dim]))
data1=np.transpose(np.reshape(targets,[N,1]))
data=[data0, data1]

tlabels=[6,9,4,8,2,0,0,0,6,5,8,7,5,1,2,1,1,2,5,7,6,4,0,9,9,0,2,5,9,4,7,7,4,3,8,9,9,5,8,3,3,6,7,6,5,5,3,2,9,1,0,8,5,4,7,2,6,5,2,3,1,1,1,2,5,6,5,8,0,3,4,9,0,0,9,6,3,5,6,6,0,0,3,6,0,6,3,2,3,6,6,4,2,1,9,2,1,7,8,3]
tlabels=np.array(tlabels).reshape(1,100)
timgs =  []
timgpath = []
id=[]
for root, dirs, files in os.walk("./test_2022"):
    for name in files:
        if name == '.DS_Store': continue
        if root.split('/')[-1]=='.ipynb_checkpoints': continue
        img = Image.open(os.path.join(root, name))
        img = np.asarray(img,dtype='float32')
        id.append(int(name.split('.')[-2]))
        timgs.append(img)
tinputs = np.array(timgs)
# ttargets = np.array(tlabels)
tdim=tinputs[0].shape[0]**2
tN=len(timgs)
tdata=np.transpose(np.reshape(tinputs, [tN,tdim]))
tlabels=np.transpose(np.reshape(tlabels, [100, 1]))
temp=tdata[:,np.argsort(id)]
test_data=[temp[:,:100],tlabels]

class layer:
    def __init__(self, last_layer_neurons, neurons, batch_size, lr=0.001, beta1=0.9, beta2=0.999, epislon=1e-8, act_func='linear'):
        self.lr = lr
        self.batch_size=batch_size
        self.dimension=neurons
        self.beta1 = beta1
        self.beta2 = beta2
        self.epislon = epislon
        self.m = 0
        self.v = 0
        self.t = 0
        self.mb = 0
        self.vb = 0
        self.tb = 0
        self.neurons=neurons
        self.hi=[]
        self.act_func=act_func
        self.act_sets={'linear':self.linear,'relu':self.relu,'leaky_relu':self.leaky_relu,'tanh':self.tanh,'sigmoid':self.sigmoid, 'softmax':self.softmax}
        self.diff_sets={'linear':self.dlinear,'relu':self.drelu,'leaky_relu':self.dleaky_relu,'tanh':self.dtanh,'sigmoid':self.dsigmoid, 'softmax':self.dlinear}
        self.weight=np.random.randn(neurons, last_layer_neurons) / np.sqrt(last_layer_neurons/2)
        self.b=np.random.uniform(0,0.1,(self.dimension,1))
        self.z=[]
    def forward(self, last):
        self.hi=last
        self.z=np.dot(self.weight, last)+self.b
        f.write(f'self.weight.shape:{self.weight.shape}\nself.b:{np.squeeze(self.b)}\n')
        return self.act_sets.get(self.act_func)(self.z)
    def backward(self, next, lr):
        self.lr=lr
        delta=next*self.diff_sets.get(self.act_func)(self.z)
        next_temp=np.transpose(self.weight)@delta
        self.t += 1
        sumw=np.zeros_like(self.weight)
        for i in range(self.batch_size):    
            sumw+=np.reshape(delta[:,i],[delta.shape[0],1])@self.hi[:,i].reshape((1, self.hi[:,i].shape[0]))
        g=sumw
        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * (g * g)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        self.weight-=self.lr* m_hat / (v_hat ** 0.5 + self.epislon)
        self.b-=0.1*self.lr*np.average(delta,axis=1, keepdims=1)
        return next_temp
    def predict(self, last, flag):
        z=self.weight@last+self.b
        return self.act_sets.get(self.act_func)(z)
    def dlinear(self, x):
        return np.ones_like(x) 
    def drelu(self, x):
        return np.where(x>0, 1, 0)
    def dleaky_relu(self, x):
        return np.where(x>=0, 1, 0.01)
    def dsigmoid(self,x):
        dsigm=x
        dsigm[np.where(dsigm<0)]=1/(1+np.exp(dsigm[np.where(dsigm<0)]))
        idx_neg=np.where(dsigm>=0)
        dsigm[idx_neg]=np.exp(-dsigm[idx_neg])/(1+np.exp(-dsigm[idx_neg]))
        return dsigm 
    def dtanh(self, x):
        return 4/(np.exp(x)+np.exp(-x))**2
    def linear(self, x):
        out=x
        return out
    def relu(self, x):
        return np.maximum(0,x)
    def leaky_relu(self, x):
        return np.maximum(0.01*x,x)
    def tanh(self, x):
        return (1-np.exp(-2*x))/(1+np.exp(-2*x))
    def sigmoid(self,x):
        sigm=x
        sigm[np.where(sigm<0)]=np.exp(sigm[np.where(sigm<0)])/(1+np.exp(sigm[np.where(sigm<0)]))
        idx_pos=np.where(sigm>=0)
        sigm[idx_pos]=1/(1+np.exp(-sigm[idx_pos]))
        return sigm 
    def softmax(self, x):
        x=x-np.max(x,axis=0)
        return np.exp(x)/np.sum(np.exp(x),axis=0,keepdims=True)

class mlp:
    def __init__(self, dim, batch_size, loss_type):
        self.lr=0.001
        self.n_layers=0
        self.n_neurons=[dim]
        self.hidden_layers=[]
        self.output_layer=None
        self.batch_size=batch_size
        self.learning_rate=10
        self.input=None
        self.output=[]
        self.is_softmax=False
        self.loss=np.ones(6)
        self.loss_type=loss_type
        self.loss_sets={'MSE':self.MSE, 'CrossEntropy':self.CrossEntropy}
        self.dloss_sets={'MSE':self.dMSE, 'CrossEntropy':self.dCrossEntropy}
    def save(self, mark):
        archi={i:[self.hidden_layers[i].act_func,self.hidden_layers[i].weight,self.hidden_layers[i].b] for i in range(self.n_layers)}
        np.save('./model_saved_image.npy',archi)
    def load_parameter(self, address):
        parameters=(np.load(address,allow_pickle=True)).item()
        for i in parameters.keys():
            self.add(parameters[i][1].shape[0], activation=parameters[i][0])
            self.hidden_layers[i].weight=parameters[i][1]
            self.hidden_layers[i].b=parameters[i][2]
    def last_layer(self, output, label):
        self.loss=np.average(self.loss_sets.get(self.loss_type)(output, label))
        if self.is_softmax:
            output[label,np.arange(0,self.batch_size)]-=1
            delta=output
        else:
            delta=self.dloss_sets.get(self.loss_type)(output, label)
        return delta
    def add(self, neurons=1, activation='linear'):
        self.n_layers+=1
        self.n_neurons.append(neurons)
        if activation=='softmax':
            self.is_softmax=True
        self.hidden_layers.append(layer( self.n_neurons[-2],neurons,self.batch_size,self.lr, act_func=activation))
    def forward(self, input):
        self.input=input
        h=self.input
        for i in range(self.n_layers):
            h=self.hidden_layers[i].forward(h)
        self.output=h
        return h
    def backward(self, label):
        next=self.last_layer(self.output, label)
        for i in range(self.n_layers-1,-1,-1):
            next= self.hidden_layers[i].backward(next, self.lr)
        return self.loss
    def predict(self, input):
        h=input
        for i in range(self.n_layers-1):
            h=self.hidden_layers[i].predict(h, False)
        i+=1
        h=self.hidden_layers[i].predict(h, True)
        if self.is_softmax:
            h=np.argmax(h,axis=0)
        return h
    def print_weight(self):
        for i in range(self.n_layers):
            print(f'weight[{i}]:{self.hidden_layers[i].weight}\nb[{i}]:{self.hidden_layers[i].b}\n')
    def MSE(self, output, label):
        return np.sum((output-label)**2, axis=0)
    def CrossEntropy(self, output, label):
        return -np.log(output[np.squeeze(label), np.arange(0,self.batch_size)]+1e-8)
    def dMSE(self, x, label):
        return 2*x-2*label
    def dCrossEntropy(self, output):
        return -1/output

def train(model,total_data,dim, N, valid, batch_size):
    i=0
    last_i=0
    loss=[]
    valid_loss=[]
    accuracy=[]
    valid_accuracy=[]
    fig_j,ax_j=plt.subplots(1,2)
    while(True):
        # 以下为进行shuffle操作，将不同轮数之间的训练批次变得不同
        permutation=np.random.permutation(total_data[0].shape[1])
        inputs=total_data[0][:,permutation]
        labels=total_data[1][:,permutation]
        loss_epoch=0
        # 以下为进行小批量的训练
        for batch_num in range(int(N/batch_size)):
            input=inputs[:,batch_num*batch_size:(batch_num+1)*batch_size]
            label=labels[:,batch_num*batch_size:(batch_num+1)*batch_size]
            output=model.forward(input)
            loss_epoch+=model.backward(label)
        loss.append((loss_epoch)/int(N/batch_size))
        

        test_size=100
        indices=np.random.choice(np.arange(0,N),size=(test_size),replace=False)
        pred_input_train=inputs[:,indices]
        pred_label_train=np.argmax(model.predict(pred_input_train),axis=0)
        accuracy.append(100*np.sum(pred_label_train==labels[:,indices])/test_size)

        pred_input_valid=valid[0]
        output_valid=model.predict(pred_input_valid)
        pred_label_valid=np.argmax(output_valid, axis=0)
        valid_loss.append(np.average(-np.log(output_valid[np.squeeze(valid[1]), np.arange(0,test_size)]+1e-8)))
        valid_accuracy.append(100*np.sum(pred_label_valid==valid[1])/test_size)
        print(f'pred_label_valid:\n{pred_label_valid}\nreal label:\n{np.squeeze(valid[1])}')
        print(f'accuracy:{accuracy[-1]}%\nvalid accuracy:{valid_accuracy[-1]}%')
        print(f'step: {i}, loss: {loss[-1]}')
        i+=1
        # 以下为存储模型参数，50轮存储一次，且如果loss下降且与上一次存储参数的轮数相差20的话也会存储
        if (valid_loss[-1]<valid_loss[0]) | (i%50==0):
            if i-last_i>20: 
                mark=f'{i}_{valid_loss[-1]:.3f}'
                mark=mark.replace('.','_')
                model.save(mark)
                last_i=i
            if(valid_loss[-1]<valid_loss[0]):
                valid_loss[0]=valid_loss[-1]

        # 以下为训练结束后进行loss和accuracy图像的绘制 
        if (i==500):
            ax_j[0].plot(np.arange(0,i),accuracy,label='train_accuracy')
            ax_j[0].set_title('train_accuracy')
            ax_j[1].plot(np.arange(0,i),loss,label='loss',c='r')
            ax_j[1].plot(np.arange(0,i),valid_loss,label='valid_loss',c='b')
            ax_j[1].set_title('loss of valid(test) set')
            plt.legend()
            plt.savefig('./img_results/acc_loss.png')
            plt.close()
            break
    f.close()
    return loss[-1]


# 以下为进行load操作,test_data为测试集数据
model=mlp(dim=4096,batch_size=30,loss_type='CrossEntropy')
model.load_parameter('./model_saved_image.npy')
pred=model.predict(test_data[0])

print(f'pred_label:\n{pred}\nreal_label:\n{np.squeeze(test_data[1])}')
print(f'accuracy:{100*np.sum(pred==np.squeeze(test_data[1]))/100}%')



# 以下为进行测试集的.csv文件的输出
pred=model.predict(tdata)
handin=np.array([id,pred])
handin=handin[:,np.argsort(id)]
dataframe=pd.DataFrame({'ID':handin[0],'GT':handin[1]})
dataframe.to_csv('./handin_load_parameters.csv',index=False)