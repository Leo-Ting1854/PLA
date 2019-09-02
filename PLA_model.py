import  numpy as np
import  matplotlib.pyplot as plt
from  sklearn import  datasets


class PLA:
    def __init__(self):
        self.W_vector=[]
        self.class_one=[]
        self.class_two=[]
    def error_calcu(self,X_vector,y,W_vector):
        error_number=0
        index=0
        while index < (X_vector.shape[0]):
            if(np.dot(X_vector[index],W_vector.T)>0):#預測成class1
                if(y[index]!=1): #預測錯誤 真實為-1
                    error_number=error_number+1
            else:
                if(y[index]==1): #預測錯誤 真實為1
                    error_number=error_number+1
            index=index+1
        return error_number

    def fit(self,X,y,pocket):

        #PLA是用向量內積來實現分類
        #例如如果只有1維特徵資料x1  根據感知機 y=ax1+b 的值如果大於0  則分成class1  反之分成class2
        #用向量來看就是(b,a) 內積(1,x1)  其中(1,x1)是根據資料固定的   (b,a)是我們要找出來的目標向量
        #所以使用PLA 會使特徵1維資料 變成1+1維 向量內積的過程
        #特徵N維資料  會變成N+1維向量內積的過程
        #例如特徵2維(x1,x2) 感知機  y=ax1+bx2+c  就是看y的值是否大於0  或小於0
        #即(c,a,b) 內積(1,x1,x2)  目標找出向量(c,a,b) 我只是為了把1擺在x1,x2前面才這樣寫的
        #假設DATA為100資料每筆資料有3維度 則DATA=100X3  W=(A,B,C) 為1X3  那DATA內積W就是   X dot W.T


        #根據這個原則 訓練時 必須把所有DATA 加上"1"項
        one_vector=np.array([1]*X.shape[0])[:,np.newaxis]
        X_vector=np.hstack((one_vector,X))#製造成功
        W_vector=np.random.rand(X_vector.shape[1])
        #製造完X_vector,W_vector
        #因為感知機1次只能分成兩類
        kind=np.unique(y)
        class_one=kind[0]
        class_two=kind[1]
        self.class_one=np.copy(class_one)
        self.class_two=np.copy(class_two)
        y_binary=np.copy(y)
        for index in range(len(y_binary)):
            if(y_binary[index]==class_one):
                y_binary[index]=1
            else:
                y_binary[index]=-1
        index=0
        last_error_time=X_vector.shape[0]
        run_time=0 #pocket 預設5000
        while index < (X_vector.shape[0]):
            error_flag=0
            if(np.dot(X_vector[index],W_vector.T)>0):#預測成1
                if(y_binary[index]!=1): #預測錯誤 真實為-1
                    error_flag=1
            else:#預測成-1
                if(y_binary[index]==1): #預測錯誤 真實為1
                    error_flag=1
            if(error_flag==1):
                W_vector=W_vector+y_binary[index]*X_vector[index]
                index=0
            else:
                index=index+1
            error_time=self.error_calcu(X_vector,y_binary,W_vector)
            if(error_time<=last_error_time):
                self.W_vector=W_vector
            last_error_time=error_time

            run_time=run_time+1
            if(pocket['pocket']==True):
                if(run_time==pocket['time']):#pocket
                    break
        print('訓練資料最後錯誤數=',error_time)
        #訓練完後得到(a,b,c)  y=ax1+bx2+c 其中y是感知器的輸出    感知器是以>0 <0  區分類別
        #所以要畫出感知器的線  就是ax1+bx2+c=0   看成是一條x1,x2的關係線:x2=-(ax1+c)/b
        #舉例來說  如果今天感知器的輸出 y是由資料(x1,x2)產生   且這個y比0大
        #就是ax1+bx2+c>0   代表x2>-(ax1+c)/b    可以看成 y值(x2)> 線(感知器):y=-(ax1+c)/b   就是這個x2 比線還高的意思(2d)
        #如果是3維輸入(x1,x2,x3)  那感知器輸出就是y=ax1+bx2+cx3+d
        #如果找到(a,b,c,d)  就代表找到感知器ax1+bx2+cx3+d=0   可以看成一個平面
        #只要在這個平面上方  ax1+bx2+cx3+d>0    代表x3>-(ax1+bx2+d)/c   就看成  z值>平面:z=-(ax1+bx2+d)/c   依此類推

        #但可視化過程不用這麼麻煩 只要用contur 就可以解決了  不用自己找線或平面方程式
        #self.binary_show(X,y)


    def predict(self,X):
        one_vector=np.array([1]*X.shape[0])[:,np.newaxis]
        X_vector=np.hstack((one_vector,X))
        y_predict=[]
        W_vector=self.W_vector
        for x_vector in X_vector:
            if(np.dot(x_vector,W_vector.T)>0):
                y_predict.append(self.class_one)
            else:
                y_predict.append(self.class_two)
        return np.array(y_predict)

    def accuaracy(self,y_predict,y):
        num=0
        for index in range(len(y_predict)):
            if (y[index]==y_predict[index]):
                num=num+1
        return num/len(y_predict)*100
    def binary_show(self,X,y):
        plt.figure()
        plt.scatter(X[:,0],X[:,1],c=y)
        x=X[:,0]
        y=X[:,1]

        x_min,x_max=np.min(x),np.max(x)
        x_point=np.linspace(x_min-0.5,x_max+0.5,100)
        y_min,y_max=np.min(y),np.max(y)
        y_point=np.linspace(y_min-0.5,y_max+0.5,100)


        X_mesh,y_mesh=np.meshgrid(x_point,y_point)
        X_ravel,y_ravel=np.ravel(X_mesh),np.ravel(y_mesh)
        locate=np.c_[X_ravel,y_ravel]
        Z=self.predict(locate).reshape(X_mesh.shape)
        plt.contour(X_mesh,y_mesh,Z)


        #plt.show()







def main():

    iris= datasets.load_iris()
    #為了可視化 特徵給兩個就好
    X=iris.data[0:100,0:2]
    y=iris.target[0:100]
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

    p=PLA()
    pocket={'pocket':True,'time':10000}
    p.fit(x_train,y_train,pocket)
    y_predict=p.predict(x_train)
    print('訓練資料的準確度',p.accuaracy(y_predict,y_train))
    y_predict=p.predict(x_test)
    print('資料的準確度',p.accuaracy(y_predict,y_test))

    if(X.shape[1]==2):
        p.binary_show(x_train,y_train)
        p.binary_show(x_test,y_test)

    plt.show()

if __name__=="__main__":
    main()