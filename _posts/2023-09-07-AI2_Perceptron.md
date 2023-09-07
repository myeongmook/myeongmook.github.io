---
layout: single
title: "Perceptron"
categories: deeplearning
tag: [Perceptron, Binary Classification, Supervised Learning]
toc: true #table of contents 생성
author_profile: false
sidebar:
   nav: "docs"
#search: false
---

> ## Perceptron
A Perceptron is an algorithm used for supervised learning of binary classifiers

퍼셉트론은 이진 분류(Binary Classification) 모델을 학습하기 위한 지도학습(Supervised Learning) 기반의 알고리즘이다.



> ## Binary Classification

\[
\begin{align*}
w &= \begin{bmatrix}
w_1 \\
w_2 \\
\vdots \\
w_m
\end{bmatrix}
, &
x &= \begin{bmatrix}
x_1 \\
x_2 \\
\vdots \\
x_m
\end{bmatrix}
\end{align*}
\]

### x : input , w : weights <br>
### z : net input, linear combination of x,z <br> <br>

$$
\Large z = \, w^T x + b
$$
<br>
### σ(z) : decision function <br>

 \begin{cases}
1 & \text{if } z \geq 0 \\
0 & \text{otherwise}
\end{cases}


$w_j = w_j + Δw_j$ <br>
$b = b + Δb$

$\Delta w_j = \eta \cdot (y^{(i)} - \hat{y}^{(i)}) \cdot x_j^{(i)}$ <br>
$\Delta b = \eta \cdot (y^{(i)} - \hat{y}^{(i)})$
<br> <br>
Threshold function <br>
Decision Boundary <br>

Perceptron 알고리즘은 모든 가중치의 합을 Threshold function, 즉 활성화함수에 넣어 본 뒤에 임계값을 넘으면 1, 그렇지 않으면 0을 출력한다.

1. weight와 bias값을 0 또는 small random number로 초기화한다.
2. 각 trainging smaple x에 대해 output ŷ을 구한다. (ŷ = 예측값)
3. weight와 bias를 업데이트한다 (z함수를 완성해나간다)
4. weight와 bias의 변화량이 0에 수렴하면 정확히 예측한 것이므로 weight와 bias는 변화하지 않는다.
5. 그렇지 않을 경우 <br>
정답 > 예측 -> +방향 <br>
정답 < 예측 -> -방향 <br>
으로 w,b 값이 조정된다.

단점 : 선형 이진 분류기인 만큼 not linearly separable한 자료들에는 적용하기 어렵다. <br>

입력데이터가 Perceptron의 가중치에 의해 융합 -> Thresholding -> Prediction




# Python Implemantation[Perceptron]

### function definition


```python
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Perceptron(object):
    """Perceptron classifier

    Parameters
    --------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset
    random_state : int
      Random number gerator seed for random weight initialization.

    Attributes
    --------------
    w_ : 1d-array
      Weigths after fitting.
    errors_ : list
      Number of misclassifications (updates) in each epoch

    """

    def __init__(self, eta = 0.01, n_iter = 50, random_state = 1): #생성자 함수[객체 만들때 무조건 한번 실행], 초기화함수
      self.eta = eta                        #learning rate
      self.n_iter = n_iter                  #iteration
      self.random_state = random_state

    def fit(self, X, y): # X[입력 Data], y[labled Data]
      """Fit training data.
      Prarameters
      -----------
      X : {array-like}, shape = [n_examples, n_features]
        Training vectors, where n_examples and n_features is the number of features
      y : array-like, shape = [n_examples]
        Target values.
      Returns
      --------
      self : object
      """
      rgen = np.random.RandomState(self.random_state)                           #rgen = random number generator (난수 발생기)
      self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = 1 + X.shape[1])      #loc[평균] ,  scale[표준편차],
      self.errors_ = []                 #class의 인스턴스 변수를 빈 리스트로 초기화

      for _ in range(self.n_iter):
        errors = 0
        for xi, target in zip(X,y):
          update = self.eta * (target - self.predict(xi))  #w,b의 변화량 구하기 위한 수식
          self.w_[1:] += update *xi
          self.w_[0] += update
          errors += int(update != 0.0)
        self.errors_.append(errors)
      return self


    def net_input(self,X):
      """Calculate net input"""
      return np.dot(X,self.w_[1:]) + self.w_[0]        # z = w^T*x + b

    def predict(self,X):
      """Return cclass label after unit step"""
      return np.where(self.net_input(X) >= 0.0, 1, -1) # sigma(z) >=0이면 1, 아니면 0[thresholding하는 부분]


```

### Training



```python
# ## Training a perceptron model on the Iris dateset
# ...
# ### Reading in the Iris data

s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
print('URL:',s)

df = pd.read_csv(s,                   # csv[Comma-Separated Values]
                 header =None,
                 encoding = 'utf-8')

df.tail() # tail() -> 하위 5개 행 반환

# ### Plotting the Iris data
# select setos and versicolor
y = df.iloc[0:100, 4].values #feature의 갯수 4개
#iris Data는 150개의 샘플이 3 클래스로 나눠져있음 즉 두개의 클래스를 가져옴
y = np.where(y == 'Iris-setosa',-1,1) # 배열의 각 원소에 대해 조건 검사 참 -> -1 거짓 -> 1, binary classification 작업에서 종종 사용

#extract sepal length and petal length
X = df.iloc[0:100, [0,2]].values

#plot data
plt.scatter(X[:50,0], X[:50,1], # 0[x축], 1[y축]
            color = 'red', marker='o', label = 'setosa'
            )
plt.scatter(X[50:100, 0], X[50:100,1],
            color='blue',marker='x',label='versicolor')

plt.xlabel('sepal length [cm]')
plt.ylabel('sepal length [cm]')
plt.legend(loc='upper left')

plt.show()
```

    URL: https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
    


    
![png](/images/2023-09-07-AI2_Perceptron/output_7_1.png)
    


## Perceptron 학습 및 epoch 수에 따른 error sample 수 확인


```python
# ### Training the perceptron model
ppn = Perceptron(eta = 0.1, n_iter= 10)

ppn.fit(X,y)

plt.plot(range(1, len(ppn.errors_) +1), ppn.errors_, marker = 'o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')

#plt.savefig('images/02_06.png',dpi=300)
plt.show()
```


    
![png](/images/2023-09-07-AI2_Perceptron/output_9_0.png)
    


## 학습한 model test


```python
# #### A function for plotting decision regions
def plot_decision_regions(X,y,classifier,resolution = 0.02) :
  #setup marker generator and color map
  markers = ('s','x','o','^','v')
  colors = ('red','blue','lightgreen','gray','cyan')
  cmap = ListedColormap(colors[:len(np.unique(y))]) #y 배열에서 추출한 클래스 수와 일치하는 색상 맵을 생성하고, 이 색상 맵을 cmap 변수에 할당

  #plot the decision surface
  x1_min, x1_max = X[:,0].min() - 1, X[:,0].max() + 1
  x2_min, x2_max = X[:,1].min() - 1, X[:,1].max() + 1
  xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                         np.arange(x2_min, x2_max, resolution))
  Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T) #모든 데이터를 vector rise 해서 classifier의 test 입력으로 넣음
  Z = Z.reshape(xx1.shape)
  plt.contourf(xx1,xx2,Z,alpha = 0.3, cmap = cmap)
  plt.xlim(xx1.min(),xx1.max())
  plt.ylim(xx2.min(),xx2.max())

  #plot class examples
  for idx, cl in enumerate(np.unique(y)): #만들어진 dicision boundary와 학습데이터가 얼마나 fit이 맞는지
    plt.scatter(x=X[y==cl, 0],
                y=X[y==cl,1],
                alpha=0.8,
                c=colors[idx],
                marker=markers[idx],
                label=cl,
                edgecolor='black')

plot_decision_regions(X,y,classifier = ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('sepal length [cm]')
plt.legend(loc = 'upper left')
#plt.savefig('images/02_08.png',dpi=300)
plt.show()
```

    <ipython-input-26-c18b8c56936a>:21: UserWarning: You passed a edgecolor/edgecolors ('black') for an unfilled marker ('x').  Matplotlib is ignoring the edgecolor in favor of the facecolor.  This behavior may change in the future.
      plt.scatter(x=X[y==cl, 0],
    


    
![png](/images/2023-09-07-AI2_Perceptron/output_11_1.png)
    

