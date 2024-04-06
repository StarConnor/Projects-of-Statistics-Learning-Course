import numpy as np
import matplotlib.pyplot as plt
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

    def __init__(self):
        self.train_data=[]
        self.test_data=[]
        self.weight=[]
        self.length=0
        self.support_vectors_=[[]]
        # Todo: initialize SVM class

    def train(self, data_train):
        self.length,_=np.shape(data_train) 
        self.length=int(0.8*self.length)
        self.train_data=data_train[:self.length]
        self.test_data=data_train[self.length:]
        X = self.train_data[:, :2]
        y = self.train_data[:, 2]
        i=0
        P=np.zeros(shape=[self.length,self.length])
        for i in range(self.length):
            for j in range(i, self.length):
                P[i,j]=y[i]*y[j]*np.dot(X[i],X[j])
                P[j,i]=P[i,j]
        P=matrix(P)
        q=matrix(-np.ones(shape=[self.length, 1]))
        G=matrix(-np.eye(self.length))
        h=matrix(np.zeros(shape=[self.length, 1]))
        A=matrix(y, (1, self.length))
        b=matrix(0.)
        sol=solvers.qp(P, q, G, h, A, b)
        alpha=np.asarray(sol['x']).ravel()
        svec1,svec2=np.argsort(alpha)[-2:]
        self.support_vectors_=np.append(self.support_vectors_,X[svec1])
        self.support_vectors_=np.append(self.support_vectors_,X[svec2],axis=0)
        self.support_vectors_=np.reshape(self.support_vectors_,(2,2))
        print("support:{}".format(self.support_vectors_))
        w=alpha[svec1]*y[svec1]*X[svec1]+alpha[svec2]*y[svec2]*X[svec2]
        b=0.5*(y[svec2]-np.dot(w,X[svec2])+y[svec1]-np.dot(w,X[svec1]))
        self.weight=np.append(w,b)
        # Todo: train model

    def predict(self, x):
        X = self.test_data[:, :2]
        y = self.test_data[:, 2]
        X = np.insert(X,2,np.ones(np.shape(self.test_data)[0]),axis=1)
        pred_y = np.dot(X,self.weight)
        pred_y = np.where(pred_y>0,1,pred_y)
        pred_y = np.where(pred_y<0,-1,pred_y)
        return pred_y 
        # Todo: predict labels
        raise NotImplementedError


if __name__ == '__main__':
    # Load dataset
    train_file = 'data/train_linear.txt'
    # test_file = 'data/test_linear.txt'
    data_train = load_data(train_file)  # dataset format [x1, x2, t], shape (N * 3)
    # data_test = load_data(test_file)

    # train SVM
    svm = SVM()
    svm.train(data_train)

    # predict
    x_train = data_train[:, :2]  # features [x1, x2]
    t_train = data_train[:, 2]  # ground truth labels
    t_train_pred = svm.predict(x_train)  # predicted labels
    x_test = data_train[svm.length:, :2]
    t_test = data_train[svm.length:, 2]
    t_test_pred = svm.predict(x_test)

    # evaluate
    acc_train = eval_acc(t_train, t_train_pred)
    acc_test = eval_acc(t_test, t_test_pred)
    print("train accuracy: {:.1f}%".format(acc_train * 100))
    print("test accuracy: {:.1f}%".format(acc_test * 100))
