import numpy as np
from matplotlib import pyplot as plt
from cvxopt import solvers, matrix


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

    def __init__(self,C,split=0.8):
        self.data=[]
        self.C=C
        self.split=split
        self.weight=[]
        self.b=0
        self.length=0
        self.support_vectors_=[]
        # Todo: initialize SVM class

    def train(self, data_train):
        self.length,_=np.shape(data_train) 
        self.length=int(self.split*self.length)
        self.data=data_train
        X = self.data[:self.length, :2]
        y = self.data[:self.length, 2]
        i=0
        P=np.zeros(shape=[self.length,self.length])
        for i in range(self.length):
            for j in range(i, self.length):
                P[i,j]=y[i]*y[j]*np.dot(X[i],X[j])
                P[j,i]=P[i,j]
        P=matrix(P)
        q=matrix(-np.ones(shape=[self.length, 1]))
        G=matrix(np.vstack((-np.eye(self.length),np.eye(self.length))))
        h=matrix(np.vstack((np.zeros(shape=[self.length, 1]),np.full([self.length,1], self.C))))
        A=matrix(y, (1, self.length))
        b=matrix(0.)
        sol=solvers.qp(P, q, G, h, A, b, verbose=False)
        alpha=np.asarray(sol['x']).ravel()
        svec=np.where(abs(alpha)>1e-1)[0]
        w=[0,0]
        b_array=[]
        for i in svec:
            #print("shape alpha:{} y:{} X:{}".format(alpha[i],y[i],X[i]))
            w+=alpha[i]*y[i]*X[i]
            self.support_vectors_=np.append(self.support_vectors_,X[i],axis=0)
        self.support_vectors_=np.reshape(self.support_vectors_,[len(svec),2])
        for i in svec:
            b_array.append(y[i]-np.dot(w,X[i]))
        self.b=np.average(b_array)
        self.weight=np.array(w)
        # Todo: train model

    def predict(self, x):
        pred_y = np.dot(x,self.weight)+self.b
        pred_y = np.where(pred_y>0,1,pred_y)
        pred_y = np.where(pred_y<0,-1,pred_y)
        return pred_y 
        # Todo: predict labels
        raise NotImplementedError
def myplot_svm(model, ax=None):
    X = model.data[:model.length, :2]
    y = model.data[:model.length,2]
    if ax is None:
        fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired, edgecolor='black', s=50)
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    XX, YY = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))
    Xmesh=np.c_[XX.flatten(), YY.flatten()]
    Z = np.dot(Xmesh, model.weight)+model.b
    Z = Z.reshape(XX.shape)
    #画数据空间中其他点的类别，用颜色区分
    ax.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    #画决策边界和 wx+b=1与wx+b=-1表示的间隔边界
    ax.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-1, 0, 1])
    #画出支持向量
    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=150,
                linewidth=1, facecolors='none', edgecolors='k')

if __name__ == '__main__':
    train_file = 'data/train_linear_intersect.txt'
    data_train = load_data(train_file)  # dataset format [x1, x2, t], shape (N * 3)

    solvers.options['show_progress']=False
    params_C = np.arange(1, 13) 
    fig, axes = plt.subplots(4, 3, figsize=(len(params_C) * 4, len(params_C)*4))
    for param_C, ax in zip(params_C, axes.ravel()):
        # C越小，对误分类的惩罚越小
        model = SVM(C=param_C)
        model.train(data_train)
        myplot_svm(model, ax=ax)
        print("C={:.6f} train_accurate rate = {}%".format(param_C, eval_acc(model.data[:model.length,2], model.predict(model.data[:model.length, :2]))*100))
        X = model.data[model.length:, :2]
        y = model.data[model.length:,2]
        print("           test_accurate rate = {}%".format(eval_acc(y, model.predict(X))*100))
        ax.set_title('C={}'.format(param_C))