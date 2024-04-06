import numpy as np
import matplotlib.pyplot as plt
import math
pi=math.pi
f=open('output.txt','w')
name='Without Adam'

class layer:
    def __init__(self, last_layer_neurons, neurons, batch_size, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, act_func='linear'):
        self.neurons=neurons
        self.batch_size=batch_size
        self.act_func=act_func
        self.weight=np.random.randn(neurons, last_layer_neurons) / np.sqrt(last_layer_neurons/2)
        self.b=np.random.uniform(0,0.1)
        self.hi=[]
        self.z=[]
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = 0
        self.v = 0
        self.t = 0
        self.act_sets={'linear':self.linear,'relu':self.relu,'leaky_relu':self.leaky_relu,'tanh':self.tanh,'sigmoid':self.sigmoid, 'softmax':self.softmax}
        self.diff_sets={'linear':self.dlinear,'relu':self.drelu,'leaky_relu':self.dleaky_relu,'tanh':self.dtanh,'sigmoid':self.dsigmoid, 'softmax':self.dlinear}
    def forward(self, last):
        self.hi=last
        self.z=self.weight@last+self.b
        return self.act_sets.get(self.act_func)(self.z)
    def backward(self, next, lr):
        self.lr=lr
        # 得到本层的delta矩阵
        delta=next*self.diff_sets.get(self.act_func)(self.z)
        # 将本层的weight与delta 相乘传入下一层中
        next_temp=np.transpose(self.weight)@delta
        # sumw是用于将每一批的所有样本的梯度进行加和
        sumw=np.zeros_like(self.weight)
        for i in range(self.batch_size):    
            sumw+=np.reshape(delta[:,i],[delta.shape[0],1])@ \
                self.hi[:,i].reshape((1, self.hi[:,i].shape[0]))
        # g为本层的损失函数对权重的梯度矩阵，以下为Adam优化算法，训练轮数更少
        sumw/=self.batch_size
        g=sumw
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * (g * g)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        self.weight-=self.lr* m_hat / (v_hat ** 0.5 + self.epsilon)
        # self.weight-=self.lr*sumw
        self.b-=0.1*self.lr*np.average(delta,axis=1, keepdims=1)
        return next_temp
        # av_delta=np.average(delta, axis=1,keepdims=1)
        # self.mb = self.beta1 * self.mb + (1 - self.beta1) * av_delta 
        # self.vb = self.beta2 * self.vb + (1 - self.beta2) * (av_delta * av_delta)
        # mb_hat = self.mb / (1 - self.beta1 ** self.t)
        # vb_hat = self.vb / (1 - self.beta2 ** self.t)
        # self.b-=self.lr* mb_hat / (vb_hat ** 0.5 + self.epsilon)
        # self.weight-=self.lr*sumw/self.batch_size
    def predict(self, last):
        z=np.dot(self.weight, last)+self.b
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
        return np.exp(x)/np.sum(np.exp(x),axis=1,keepdims=True)

class mlp:
    def __init__(self, dim, batch_size, loss_type):
        self.n_neurons=[dim]
        self.hidden_layers=[]
        self.n_layers=0
        self.batch_size=batch_size
        self.lr=0.001
        self.output=[]
        self.loss=np.ones(6)
        self.loss_type=loss_type
        self.loss_sets={'MSE':self.MSE, 'CrossEntropy':self.CrossEntropy}
        self.dloss_sets={'MSE':self.dMSE, 'CrossEntropy':self.dCrossEntropy}
    def save(self, mark):
        archi={i:[self.hidden_layers[i].act_func,self.hidden_layers[i].weight,self.hidden_layers[i].b] for i in range(self.n_layers)}
        np.save('./model_saved_sin.npy',archi)
    def load_parameter(self, address):
        parameters=(np.load(address,allow_pickle=True)).item()
        for i in parameters.keys():
            self.add(parameters[i][1].shape[0], activation=parameters[i][0])
            self.hidden_layers[i].weight=parameters[i][1]
            self.hidden_layers[i].b=parameters[i][2]
    def add(self, neurons=1, activation='linear'):
        self.n_layers+=1
        self.n_neurons.append(neurons)
        self.hidden_layers.append(layer(self.n_neurons[-2],neurons,self.batch_size,self.lr, act_func=activation))
    def forward(self, input):
        h=input
        for i in range(self.n_layers):
            h=self.hidden_layers[i].forward(h)
        self.output=h
        return h
    def predict(self, input):
        h=input
        for i in range(self.n_layers):
            h=self.hidden_layers[i].predict(h)
        return h
    def output_layer(self, output, label):
        self.loss=np.average(self.loss_sets.get(self.loss_type)(output, label))
        if self.loss_type=='softmax':
            output[label,np.arange(0,self.batch_size)]-=1
            delta=output
        else:
            delta=self.dloss_sets.get(self.loss_type)(output, label)
        return delta
    def backward(self, label):
        next=self.output_layer(self.output, label)
        for i in range(self.n_layers-1,-1,-1):
            next= self.hidden_layers[i].backward(next, self.lr)
        return self.loss
    def print_weight(self):
        for i in range(self.n_layers):
            print(f'weight[{i}]:{self.hidden_layers[i].weight}\nb[{i}]:{self.hidden_layers[i].b}\n')
    def MSE(self, output, label):
        return np.sum((output-label)**2, axis=0)
    def CrossEntropy(self, output, label):
        return -np.dot(label, np.log(output))
    def dMSE(self, x, label):
        return 2*x-2*label
    def dCrossEntropy(self, x, label):
        return -label/x


def create_model(dim, batch_size):
    model = mlp(dim, batch_size,'MSE')
    model.add(10,activation = 'tanh')
    model.add(100,activation = 'tanh')
    model.add(10,activation = 'tanh')
    model.add(dim,activation = 'linear')
    # model.compile(optimizer = 'adam',
    #              loss = 'sparse_categorical_crossentropy',
    #              metrics = ['accuracy'])
    return model

def train(model,total_data, N):
    i=0
    loss=[]
    epochs=np.arange(0, 300, 20)
    figloss, axloss=plt.subplots(1,2, figsize=(12, 6))
    fig, ax=plt.subplots(3,5,figsize=(4*5,4*3))
    ax_ravel=ax.ravel()
    j=0
    
    test=np.linspace(-pi,pi,400).reshape(1,400)
    total_data[0]=(total_data[0]+pi)/(2*pi)
    while(True):
        permutation=np.random.permutation(total_data[0].shape[0])
        total_data[0]=total_data[0][permutation]
        total_data[1]=total_data[1][permutation]
        permutation=np.random.permutation(total_data[0].shape[1])
        total_data[0]=total_data[0][:,permutation]
        total_data[1]=total_data[1][:,permutation]
        loss_epoch=0
        for batch_num in range(int(N/model.batch_size)):
            input=total_data[0,:,batch_num*model.batch_size:(batch_num+1)*model.batch_size]
            label=total_data[1,:,batch_num*model.batch_size:(batch_num+1)*model.batch_size]
            output=model.forward(input)
            loss_epoch+=model.backward(label)
        loss.append((loss_epoch)/int(N/model.batch_size))
        if i in epochs:
            ax_ravel[j].plot(np.squeeze(test),np.sin(np.squeeze(test)),c='r',label='real')
            pred=model.predict((test+pi)/(2*pi))
            ax_ravel[j].plot(np.squeeze(test),np.squeeze(pred),linestyle='--',c='b',label='predict')
            ax_ravel[j].legend()
            ax_ravel[j].set_title(f'epoch:{i} fitting')
            j+=1
        if (i%100==0):
            print(f'step: {i}, loss: {loss[-1]}')
        i+=1
        if (loss[-1]<1e-4) | (i==2000):
            plt.savefig('/mnt/c/Users/Connor/Desktop/lab2/sin'+name+'.png')
            plt.close()
            axloss[0].plot(np.arange(0,i),loss)
            # axloss.text(i/2, 0.2, f'loss:{np.reshape(loss[-5:],(5,1))}',)
            axloss[0].set_title('Loss in Regression of sin(x)('+name+')')
            axloss[0].set_ylim([0, 1])
            axloss[0].set_xlabel('epochs')
            axloss[0].set_ylabel('loss')
            axloss[1].plot(np.arange(0,i),np.log10(loss))
            # axloss[1].set_xscale('log')
            axloss[1].set_title('Loss in Regression of sin(x)(log mode)('+name+')')
            axloss[1].set_ylim([-4, 0])
            axloss[1].set_xlabel('epochs')
            axloss[1].set_ylabel('log(loss)')
            plt.savefig('/mnt/c/Users/Connor/Desktop/lab2/loss'+name+'.png')
            plt.close('all')
            break
    f.close()
    return loss[-1]
    
e=[]
dim=1
N=400
for i in range(dim*N):
    e.append(np.random.uniform(-np.pi, np.pi))
e=np.array(e).reshape((dim, N))
X_train=np.array((e, np.sin(e)))
data=X_train
model=create_model(dim, batch_size=80)
loss_train=train(model, data,N)
model.save('1')

e=[]
dim=1
N=20
for i in range(dim*N):
    e.append(np.random.uniform(-np.pi, np.pi))
e=np.array(e).reshape((dim, N))
X_testo=np.array((e, np.sin(e)))
X_test=np.array((e, np.sin(e)))
X_test[0]=(X_test[0]+pi)/(2*pi)

model=mlp(dim=4096,batch_size=30,loss_type='CrossEntropy')
model.load_parameter('./model_saved_sin.npy')
predy=model.predict(X_test[0,:])

loss_test = np.average(model.MSE(X_test[1,:], predy))
print(f'train MSE: {loss_train}')
print(f'test MSE: {loss_test}')
figtest,axtest=plt.subplots(1,2,figsize=(12, 6))
axtest[0].scatter(np.squeeze(X_testo[0]),np.squeeze(X_test[1]),c='r',label='actual')
axtest[0].scatter(np.squeeze(X_testo[0]),np.squeeze(predy),c='b',label='predict')
axtest[0].set_title('Random selected Points and their Pediction('+name+')')
figtest.suptitle(name+f'\ntrain MSE:{loss_train}  test MSE: {loss_test}\n',font={'size':18})

test=np.linspace(-pi,pi,400).reshape(1,400)
axtest[1].plot(np.squeeze(test),np.sin(np.squeeze(test)),c='r',label='real')
pred=model.predict((test+pi)/(2*pi))
axtest[1].plot(np.squeeze(test),np.squeeze(pred),linestyle='--',c='b',label='predict')
axtest[1].set_title('400 Uniform Points and their Prediction ('+name+')')
axtest[1].legend()
plt.savefig('/mnt/c/Users/Connor/Desktop/lab2/fit'+name+'.png')