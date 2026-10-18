import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# sütunlar: [yaş (yıl), km, motor hacmi (cc)]
X_train = np.array([
    [ 3,  49000, 1600], [ 6, 122000, 1800], [ 1,  12000, 1000], [10, 187000, 1400],
    [ 7,  68000, 1400], [ 5,  73000, 1400], [ 5,  50000, 2000], [ 6, 112000, 1600],
    [ 5,  85000, 2000], [ 5,  90000, 1600], [ 8,  87000, 1000], [ 7, 115000, 1800],
    [ 5,  61000, 2000], [ 5,  62000, 1800], [ 6,  95000, 2200], [ 4,  68000, 1400],
    [ 4,  59000, 2000], [ 7,  83000, 1000], [ 5,  56000, 1000], [ 1,  19000, 2000],
    [ 4,  64000, 2200], [ 7, 116000, 1600], [ 5, 108000, 2000], [ 5,  56000, 1400],
])
y_train = np.array([561, 343, 621, 43, 338, 417, 568, 346, 490, 405, 261, 300,
                    507, 548, 439, 508, 574, 298, 406, 770, 574, 348, 431, 466])  # bin TL

X_test = np.array([
    [10, 164000, 2000], [10, 161000, 1800], [ 5,  36000, 1000],
    [ 7,  54000, 1400], [ 8,  76000, 1200], [ 6,  86000, 2000],
])
y_test = np.array([144, 127, 409, 391, 317, 437])  # bin TL

#burada parametreleri tanımlıyoruz notasyon tarzı
# X_train = x katsayıları
# ypred =w*x +b
# yaptıgımız muabbet graident descentte eniyi w b yi bulmak
# x ve y train ile eğitiyoruz ve x y test ile hata farkını yakalıyoruz bunu da azaltmak için cost function kullanıyoruz
#m=satır sayısı
#n=sutun satyıyı
#
#
m=X_train.shape[0]#satır 
n=X_train.shape[1]#sutun
w=np.zeros(n)
b=0.0
iterations=1000
y_pred=np.dot(X_train,w)+b  
   #moodeli yazıyorum bütün xtrain değeri için çarpıp toplasın tek değer cıkarsın 
#modeli her yerde tekrar tekrar yazmak istemiyorum global tanımladım ondan dolayı
learnin_rate=0.0004 #bunu mühendis değiştirir en optimali bulmak için bi bakıcam simdilik 2 verdim


def compute_cost (y_train,m,y_pred):#model için olan parametrelere gerek yok vermedim ondan
    error=y_pred-y_train
    cost=(1/(2*m))*np.sum(np.square(error))
    return cost

    
def compute_gradient(m,y_train,y_pred,X_train):
  
    dj_dw=(1/m)*np.dot(X_train.T,y_pred-y_train)
    dj_db=(1/m)*np.sum(y_pred-y_train)
    return dj_dw,dj_db

def gradient_descent(w,b,m,y_train,y_pred,X_train,iterations,learnin_rate):
   
   for i in range(iterations):
    y_pred=np.dot(X_train,w)+b 
    dj_dw,dj_db=compute_gradient(m,y_train,y_pred,X_train)
    w=w-learnin_rate*dj_dw
    b=b-learnin_rate*dj_db

    y_pred=np.dot(X_train,w)+b 
   return w,b 

#grafikler
plt.scatter(X_train[:,1],y_train)
plt.xlabel("kilometre")
plt.ylabel("fiyat")
w,b=gradient_descent(w,b,m,y_train,y_pred,X_train,iterations,learnin_rate)#modeli eğittim
y_pred=np.dot(X_train,w)+b  #SON DEĞERLER İLE YAPILAN TAHMİNLER 

order = np.argsort(X_train[:, 1])
plt.plot(X_train[order, 1], y_pred[order], color="red")
plt.show()



   



         
