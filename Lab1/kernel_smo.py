import random
import numpy as np
import math
from matplotlib import pyplot as plt

def myplot_svm(model, data_test, ax=None):
    X = data_test[:, :2]
    y = data_test[:,2]
    # X = data_train[:, :2]
    # y = data_train[:,2]
    if ax is None:
        fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired, edgecolor='black', s=50)
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    XX, YY = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))
    Xmesh=np.c_[XX.flatten(), YY.flatten()]
    Z = model.map(Xmesh)
    Z = Z.reshape(XX.shape)
    #画数据空间中其他点的类别，用颜色区分
    ax.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    #画决策边界和 wx+b=1与wx+b=-1表示的间隔边界
    ax.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-1, 0, 1])
    #画出支持向量
    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=150,
                linewidth=1, facecolors='none', edgecolors='k')
# Load dataset
def load_data(fname):
    with open(fname, 'r') as f:
        data = []
        line = f.readline()
        for line in f:
            line = line.strip().split()
            x1 = float(line[0])
            x2 = float(line[1])
            t = int(line[2])
            data.append([x1, x2, t])
        return np.array(data)


# Calculate classification accuracy
def eval_acc(label, pred):
    return np.sum(label == pred) / len(pred)


# Visualization
def show_data(data):
    fig, ax = plt.subplots()
    cls = data[:, 2]
    ax.scatter(data[:, 0][cls==1], data[:, 1][cls==1])
    ax.scatter(data[:, 0][cls==-1], data[:, 1][cls==-1])
    ax.grid(False)
    fig.tight_layout()
    plt.show()

class SVM():

    def __init__(self,C,split=1.):
        self.train_data=[]
        self.X=[]
        self.y=[]
        self.alpha=[]
        self.kappa=[]
        self.C=C
        self.sigma2=1
        self.split=split
        self.weight=[]
        self.b=0
        self.length=0
        self.support_vectors_=[]
        self.stop=100
        self.ac=1e-5
        # Todo: initialize SVM class

    def train(self, data_train, f, ax):
        self.length,_=np.shape(data_train) 
        self.length=int(self.split*self.length)
        self.train_data=data_train[:self.length]
        self.X = self.train_data[:, :2]
        self.y = np.squeeze(self.train_data[:, 2])
        X=self.X
        y=self.y
        self.kappa=np.zeros(shape=[self.length,self.length])
        for i in range(self.length):
            for j in range(i, self.length):
                self.kappa[i,j]=math.exp(-(np.linalg.norm(self.X[i]-self.X[j]))**2)/(2*self.sigma2)
                self.kappa[j,i]=self.kappa[i,j]
        self.alpha=np.zeros(self.length)
        j=1
        acc=[]
        while(True):
            # catei(i=1，2，3)分别对应着alpha=0, 0<alpha<C, alpha=C三种情况下的alpha的索引
            cate2=np.where(((self.alpha)>=self.ac) & ((self.C-self.alpha)>=self.ac))[0]
            cate1=np.where((self.alpha)<self.ac)[0]
            cate3=np.where((self.C-self.alpha)<self.ac)[0]
            # kkt计算违背KKT条件的值的大小，kkt值越大，对应的坐标点违背KKT的程度越大
            kkt=y[np.arange(0, self.length)]*self.decision(np.arange(0, self.length))
            kkt[cate1]=1-kkt[cate1]
            kkt[cate2]=abs(kkt[cate2]-1)
            kkt[cate3]=kkt[cate3]-1
            # violatei(i=1,2,3)存储着违背KKT条件的点的索引
            violate1=cate1[np.where(kkt[cate1]>self.ac)]
            violate2=cate2[np.where(kkt[cate2]>self.ac)]
            violate3=cate3[np.where(kkt[cate3]>self.ac)]
            violate=np.concatenate((violate1,violate2,violate3))
            # 首先遍历0<alpha<C情况下的点，再考虑alpha=0和alpha=C
            if len(violate2)==0:
                vio13=np.concatenate((violate1,violate3))
                idx_alpha1=vio13[np.argsort(kkt[vio13])[-1]]
            else:
                idx_alpha1=violate2[np.argsort(kkt[violate2])[-1]]
            E1=self.decision(idx_alpha1)-y[idx_alpha1]
            # 根据alpha_1的点找到距离alpha_1最远的alpha_2
            idx_alpha2=violate[np.argsort(abs(E1-(self.decision(violate)-y[violate])))[-1]]
            if idx_alpha2==idx_alpha1:
                if len(violate)>=2: 
                    idx_alpha2=violate[np.argsort(abs(E1-(self.decision(violate)-y[violate])))[-2]]
                else:
                    search_range=np.argsort(kkt)[-5:]
                    idx_alpha2=np.argsort(abs(E1-(self.decision(search_range)-y[search_range])))[-1]
            # 下面的程序对alpha_1和alpha_2进行更新操作
            E2=self.decision(idx_alpha2)-y[idx_alpha2]
            a_2=self.alpha[idx_alpha2]
            y_2=y[idx_alpha2]
            deno=self.kappa[idx_alpha1,idx_alpha1]+self.kappa[idx_alpha2,idx_alpha2]-2*self.kappa[idx_alpha1,idx_alpha2]
            alpha_star=a_2+y_2*(E1-E2)/deno
            if y[idx_alpha1]==y[idx_alpha2]:
                H=min(self.C, self.alpha[idx_alpha2]+self.alpha[idx_alpha1])
                L=max(0, self.alpha[idx_alpha2]+self.alpha[idx_alpha1]-self.C)
            else:
                H=min(self.C, self.C+self.alpha[idx_alpha2]-self.alpha[idx_alpha1])
                L=max(0, self.alpha[idx_alpha2]-self.alpha[idx_alpha1])
            a2_old=self.alpha[idx_alpha2]
            a1_old=self.alpha[idx_alpha1]
            if alpha_star>H:
                a2_new=H
                self.alpha[idx_alpha2]=H
            elif alpha_star<L:
                a2_new=L
                self.alpha[idx_alpha2]=L
            else:  
                a2_new=alpha_star
                self.alpha[idx_alpha2]=alpha_star
            self.alpha[idx_alpha1]=a1_old+y[idx_alpha1]*y[idx_alpha2]*(a2_old-a2_new)
            a1_new=self.alpha[idx_alpha1]
            b1_new=-E1-y[idx_alpha1]*self.kappa[idx_alpha1,idx_alpha1]*(a1_new-a1_old)-y[idx_alpha2]*self.kappa[idx_alpha2,idx_alpha1]*(a2_new-a2_old)+self.b
            b2_new=-E2-y[idx_alpha1]*self.kappa[idx_alpha1,idx_alpha2]*(a1_new-a1_old)-y[idx_alpha2]*self.kappa[idx_alpha2,idx_alpha2]*(a2_new-a2_old)+self.b
            # 对b进行更新，加上一个噪声使得程序不会陷入局部最优解，从而可以跳出来并达到全局最优解
            self.b=(b1_new+b2_new)/2+random.uniform(0, 1e-3)

            if abs(np.dot(self.alpha,y))<self.ac:
                kkt=y*self.decision(np.arange(0,self.length))
                temp1=np.count_nonzero(kkt[np.where((self.alpha)<self.ac)]<1)
                temp2=np.count_nonzero(kkt[np.where((self.C-self.alpha)<self.ac)]>1)
                temp3=np.count_nonzero(abs(kkt[np.where(((self.C-self.alpha)>self.ac)&((self.alpha)>self.ac))]-1)>self.ac)
                acc.append(100*(temp1+temp2+temp3)/self.length)
                f.write(f'epoch:{j}, violate kkt:{acc[-1]}% with 1:{temp1},2:{temp3},3:{temp2}\n\n')
            if acc[-1]<self.ac:
                x=np.arange(0,j)
                ax.plot(x,acc,label=f'C={self.C}')
                ax.legend(loc='best')
                f.write(f'final alpha:{self.alpha}')
                break
            j+=1


    def predict(self, x):
        support=np.where(self.alpha>0)[0]
        self.support_vectors_=self.X[support]
        para=self.alpha*self.y
        pred_y=0
        for i in support:
            #print(f'i:{i}')
            kappa=np.exp(-(np.linalg.norm(x-self.X[i],axis=1,keepdims=True))**2)/(2*self.sigma2)
            pred_y+=kappa*para[i]
        pred_y+=self.b
        pred_y = np.where(pred_y>0,1,pred_y)
        pred_y = np.where(pred_y<0,-1,pred_y)
        return np.squeeze(pred_y) 
        # Todo: predict labels
        raise NotImplementedError
    def map(self, x):
        support=np.where(self.alpha>0)[0]
        self.support_vectors_=self.X[support]
        para=self.alpha*self.y
        pred_y=0
        for i in support:
            #print(f'i:{i}')
            kappa=np.exp(-(np.linalg.norm(x-self.X[i],axis=1,keepdims=True))**2)/(2*self.sigma2)
            pred_y+=kappa*para[i]
        pred_y+=self.b
        return np.squeeze(pred_y) 
    def decision(self, idx):
        return np.dot(self.alpha*self.y, self.kappa[:,idx])+self.b

train_file = 'data/train_kernel.txt'
test_file = 'data/test_kernel.txt'
data_train = load_data(train_file)  # dataset format [x1, x2, t], shape (N * 3)
data_test = load_data(test_file)  # dataset format [x1, x2, t], shape (N * 3)

with open('output.txt','w') as f:   
    params_C = np.arange(1, 21)
    fig1, ax1 = plt.subplots()
    ax1.scatter(data_train[:, 0], data_train[:, 1], c=data_train[:,2], zorder=10, cmap=plt.cm.Paired, edgecolor='black', s=50)
    fig2, ax2 = plt.subplots()
    ax2.scatter(data_test[:, 0], data_test[:, 1], c=data_test[:,2], zorder=10, cmap=plt.cm.Paired, edgecolor='black', s=50)
    fig, axes = plt.subplots(5, 4, figsize=(len(params_C) * 4, 4*len(params_C)))
    fig4, axes4 = plt.subplots(5, 4, figsize=(len(params_C) * 4, 4*len(params_C)))
    acc_test=np.asarray([])
    acc_train=np.asarray([])
    for param_C, ax, ax4 in zip(params_C, axes.ravel(), axes4.ravel()):
        model = SVM(C=param_C)
        model.train(data_train,f, ax4)
        acc_test=np.append(acc_test,eval_acc(data_test[:,2],model.predict(data_test[:,:2])))
        acc_train=np.append(acc_train,eval_acc(data_train[:,2],model.predict(data_train[:,:2])))
        print(f'accurate:{acc_test[-1]}')
        #if param_C==2:
            #print(model.alpha)
        myplot_svm(model, data_train, ax)
        ax.set_title('C={}'.format(param_C))
    f.close()
    fig3, ax3 = plt.subplots()
    ax3.plot(params_C,acc_test,c='red',label='Test')
    ax3.plot(params_C,acc_train,c='red',label='Train',linestyle='--')
    ax3.scatter(params_C,acc_test,c='red')
    ax3.scatter(params_C,acc_train,c='red')
    ax3.legend(loc='best')
    ax3.set_xlabel('C')
    ax3.set_ylabel('Accurate Rate')
    ax3.set_title('Graph of the accurate rate w.r.t parameter C')
    plt.show()