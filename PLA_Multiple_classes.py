import  numpy as np
import  matplotlib.pyplot as plt
from  sklearn import  datasets
from mpl_toolkits.mplot3d import  Axes3D

class PLA:
    def __init__(self):
        self.W_vector=[]
        self.class_one=[]
        self.class_two=[]
        self.kind=[]
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
        #用向量來看就是(a,b) 內積(x1,1)  其中(x1,1)是根據資料固定的   (a,b)是我們要找出來的目標向量

        #所以使用PLA 會使特徵1維資料 變成1+1維 向量內積的過程
        #特徵N維資料  會變成N+1維向量內積的過程
        #例如特徵2維(x1,x2) 感知機  y=ax1+bx2+c  就是看y的值是否大於0  或小於0
        #即(a,b,c) 內積(x1,x2,1)  目標找出向量(a,b,c)

        #假設DATA為100資料每筆資料有3維度 則DATA=100X3
        #就要找到平面ax1+bx2+cx3+d=0
        #要先把Data尾端添加 "1" 變成(x1,x2,x3,1)  及 W=(A,B,C,D) 為1X4
        #那DATA內積W就是   X dot W.T


        #根據這個原則 訓練時 必須把所有DATA 加上"1"項
        one_vector=np.array([1]*X.shape[0])[:,np.newaxis]
        X_vector=np.hstack((X,one_vector))#製造成功
        #製造完X_vector,W_vector
        #因為感知機1次只能分成兩類
        kind=np.unique(y)
        self.kind=kind
        y_binary=np.copy(y)


        for sort in kind:  #ex kind=[0,1,2]  then need 3 perceptron
            y_binary=np.copy(y)
            self.W_vector.append([])
            W_vector=np.zeros((1,X_vector.shape[1]))
            for index in range(len(y_binary)):
                if(y_binary[index]==sort):#正在分的這一類
                    y_binary[index]=1
                else:#其他類
                    y_binary[index]=-1

            #PLA算法是利用 這一類=1  其他類=-1 來分類  所以只能有1跟-1 存在
            index=0
            last_error_time=X_vector.shape[0]
            run_time=0 #pocKet
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
                    self.W_vector[sort]=np.copy(W_vector)
                last_error_time=error_time

                run_time=run_time+1
                if(pocket['pocket']==True):
                    if(run_time==pocket['time']):#pocket 取準確度最高的
                        break #PLA算法

            print('第',sort+1,'個感知器','訓練資料最後錯誤數=',error_time)


        #假設是 2維的特徵 代表data處理後是(x1,x2,1)
        #訓練完後得到W=(a,b,c) y=ax1+bx2+c 其中y是感知器的輸出    感知器是以>0 <0  區分類別
        #所以要畫出感知器的線  就是ax1+bx2+c=0   看成是一條x1,x2的關係線:x2=-(ax1+c)/b
        #舉例來說  如果今天感知器的輸出 y是由資料(x1,x2)產生   且這個y比0大
        #就是ax1+bx2+c>0   代表x2>-(ax1+c)/b    可以看成 y值(x2)> 線(感知器):y=-(ax1+c)/b   就是這個x2 比線還高的意思(2d)
        #如果是3維輸入(x1,x2,x3)  那感知器輸出就是y=ax1+bx2+cx3+d
        #如果找到(a,b,c,d)  就代表找到感知器ax1+bx2+cx3+d=0   可以看成一個平面
        #只要在這個平面上方  ax1+bx2+cx3+d>0    代表x3>-(ax1+bx2+d)/c   就看成  z值>平面:z=-(ax1+bx2+d)/c   依此類推

        #如果是2d可視化過程只要用contur 就可以解決了  不用自己找線或平面方程式
        #self.binary_show(X,y)
        #如果是3d 可視化過程就要找出z=ax+by+c 的方程式了
        #W=[w0,w1,w2,w3]  X=[x1,x2,x3,1]
        #w0x1+w1x2+w2x3+w3*1=0
        #x3=-(w0x1+w1x2+w3)/w2  #即可畫出平面 注意:w2


    def predict(self,X):
        one_vector=np.array([1]*X.shape[0])[:,np.newaxis]
        X_vector=np.hstack((X,one_vector))
        y_predict=[]
        W_vector=self.W_vector

        for x_vector in X_vector:
            sort=np.copy(self.kind)*0
            sort_done_flag=0
            for W_index in range (len(W_vector)):#依序套入感知器
                if(np.dot(x_vector,W_vector[W_index].T)>0):#如果今天是分成這一類
                    sort[W_index]+=1
                    sort_done_flag=1
            if(sort_done_flag==0):
                y_predict.append(np.random.randint(0,self.kind[len(self.kind)-1]))
            else:
                y_predict.append(np.argmax(sort))




        return np.array(y_predict)

    def accuaracy(self,y_predict,y):
        num=0

        for index in range(len(y_predict)):
            if (y[index]==y_predict[index]):
                num=num+1
        return num/len(y_predict)*100


    def _2D_3D_show(self,X,y):
        if(X.shape[1]==3):
            fig=plt.figure()#產生視窗
            ax=Axes3D(fig)#產生3D 軸
            ax.scatter(X[:,0],X[:,1],X[:,2],c=y)
            ##note
            #W=[w0,w1,w2,w3]
            #x3=-(w0x1+w1x2+w3)/w2  #即可畫出平面 注意:w2
            index=0
            for W in self.W_vector:
                w0,w1,w2,w3=W[0],W[1],W[2],W[3]
                x1=X[:,0]
                x2=X[:,1]
                x_min,x_max=np.min(x1),np.max(x1)
                x_point=np.linspace(x_min-0.5,x_max+0.5,100)
                y_min,y_max=np.min(x2),np.max(x2)
                y_point=np.linspace(y_min-0.5,y_max+0.5,100)
                X_mesh,y_mesh=np.meshgrid(x_point,y_point)
                x3=-(X_mesh*w0+y_mesh*w1+w3)/w2
                if(index==0):
                    surf =ax.plot_surface(X_mesh, y_mesh,x3,cmap="summer")
                elif(index==1):
                    surf =ax.plot_surface(X_mesh, y_mesh,x3,cmap="cool")
                else:
                    surf =ax.plot_surface(X_mesh, y_mesh,x3)
                index=index+1
        elif(X.shape[1]==2):
            fig=plt.figure()#產生視窗
            plt.scatter(X[:,0],X[:,1],c=y)
            #note
            #W=[w0,w1,w2]   data=[x1,x2,1]   x1w0+x2w1+w2=0  x2=-(x1w0+w2)/w1
            #x2=-(x1w0+w2)/w1  #即可畫出分割線 注意:w1
            for W in self.W_vector:
                w0,w1,w2=W[0],W[1],W[2]
                x=X[:,0]
                x_min,x_max=np.min(x),np.max(x)
                x_point=np.linspace(x_min-0.5,x_max+0.5,100)
                y=-(w0*x_point+w2)/w1
                plt.plot(x_point,y)
        else:
            print('# WARNING:  ')
            print('not 2D or 3D  data')


def main():

    iris= datasets.load_iris()
    from sklearn.preprocessing import  scale
    # X=iris.data[0:100,[0,1]]
    # X=scale(X)
    # y=iris.target[0:100]
    # X,y=datasets.make_blobs(200,n_features=3)
    digit=datasets.load_digits()
    X=digit.data

    y=digit.target

    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

    p=PLA()
    pocket={'pocket':True,'time':5000}
    p.fit(x_train,y_train,pocket)
    y_predict=p.predict(x_train)

    print('訓練資料的準確度',p.accuaracy(y_predict,y_train))
    y_predict=p.predict(x_test)
    print('測試資料的準確度',p.accuaracy(y_predict,y_test))
    #
    p._2D_3D_show(x_train,y_train)
    plt.title('train_data')
    p._2D_3D_show(x_test,y_test)
    plt.title('test_data')
    plt.show()

if __name__=="__main__":
    main()
