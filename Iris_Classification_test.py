
# coding: utf-8

# ### Required libraries
# 

# In[196]:


import numpy as np
import matplotlib.pyplot as plt 
from sklearn import preprocessing


# ### Data

# In[197]:


data = np.array([['5.1', '3.5', '1.4', '0.2', 'Iris-setosa'],
       ['4.9', '3.0', '1.4', '0.2', 'Iris-setosa'],
       ['4.7', '3.2', '1.3', '0.2', 'Iris-setosa'],
       ['4.6', '3.1', '1.5', '0.2', 'Iris-setosa'],
       ['5.0', '3.6', '1.4', '0.2', 'Iris-setosa'],
       ['5.4', '3.9', '1.7', '0.4', 'Iris-setosa'],
       ['4.6', '3.4', '1.4', '0.3', 'Iris-setosa'],
       ['5.0', '3.4', '1.5', '0.2', 'Iris-setosa'],
       ['4.4', '2.9', '1.4', '0.2', 'Iris-setosa'],
       ['4.9', '3.1', '1.5', '0.1', 'Iris-setosa'],
       ['5.4', '3.7', '1.5', '0.2', 'Iris-setosa'],
       ['4.8', '3.4', '1.6', '0.2', 'Iris-setosa'],
       ['4.8', '3.0', '1.4', '0.1', 'Iris-setosa'],
       ['4.3', '3.0', '1.1', '0.1', 'Iris-setosa'],
       ['5.8', '4.0', '1.2', '0.2', 'Iris-setosa'],
       ['5.7', '4.4', '1.5', '0.4', 'Iris-setosa'],
       ['5.4', '3.9', '1.3', '0.4', 'Iris-setosa'],
       ['5.1', '3.5', '1.4', '0.3', 'Iris-setosa'],
       ['5.7', '3.8', '1.7', '0.3', 'Iris-setosa'],
       ['5.1', '3.8', '1.5', '0.3', 'Iris-setosa'],
       ['5.4', '3.4', '1.7', '0.2', 'Iris-setosa'],
       ['5.1', '3.7', '1.5', '0.4', 'Iris-setosa'],
       ['4.6', '3.6', '1.0', '0.2', 'Iris-setosa'],
       ['5.1', '3.3', '1.7', '0.5', 'Iris-setosa'],
       ['4.8', '3.4', '1.9', '0.2', 'Iris-setosa'],
       ['5.0', '3.0', '1.6', '0.2', 'Iris-setosa'],
       ['5.0', '3.4', '1.6', '0.4', 'Iris-setosa'],
       ['5.2', '3.5', '1.5', '0.2', 'Iris-setosa'],
       ['5.2', '3.4', '1.4', '0.2', 'Iris-setosa'],
       ['4.7', '3.2', '1.6', '0.2', 'Iris-setosa'],
       ['4.8', '3.1', '1.6', '0.2', 'Iris-setosa'],
       ['5.4', '3.4', '1.5', '0.4', 'Iris-setosa'],
       ['5.2', '4.1', '1.5', '0.1', 'Iris-setosa'],
       ['5.5', '4.2', '1.4', '0.2', 'Iris-setosa'],
       ['4.9', '3.1', '1.5', '0.1', 'Iris-setosa'],
       ['5.0', '3.2', '1.2', '0.2', 'Iris-setosa'],
       ['5.5', '3.5', '1.3', '0.2', 'Iris-setosa'],
       ['4.9', '3.1', '1.5', '0.1', 'Iris-setosa'],
       ['4.4', '3.0', '1.3', '0.2', 'Iris-setosa'],
       ['5.1', '3.4', '1.5', '0.2', 'Iris-setosa'],
       ['5.0', '3.5', '1.3', '0.3', 'Iris-setosa'],
       ['4.5', '2.3', '1.3', '0.3', 'Iris-setosa'],
       ['4.4', '3.2', '1.3', '0.2', 'Iris-setosa'],
       ['5.0', '3.5', '1.6', '0.6', 'Iris-setosa'],
       ['5.1', '3.8', '1.9', '0.4', 'Iris-setosa'],
       ['4.8', '3.0', '1.4', '0.3', 'Iris-setosa'],
       ['5.1', '3.8', '1.6', '0.2', 'Iris-setosa'],
       ['4.6', '3.2', '1.4', '0.2', 'Iris-setosa'],
       ['5.3', '3.7', '1.5', '0.2', 'Iris-setosa'],
       ['5.0', '3.3', '1.4', '0.2', 'Iris-setosa'],
       ['7.0', '3.2', '4.7', '1.4', 'Iris-versicolor'],
       ['6.4', '3.2', '4.5', '1.5', 'Iris-versicolor'],
       ['6.9', '3.1', '4.9', '1.5', 'Iris-versicolor'],
       ['5.5', '2.3', '4.0', '1.3', 'Iris-versicolor'],
       ['6.5', '2.8', '4.6', '1.5', 'Iris-versicolor'],
       ['5.7', '2.8', '4.5', '1.3', 'Iris-versicolor'],
       ['6.3', '3.3', '4.7', '1.6', 'Iris-versicolor'],
       ['4.9', '2.4', '3.3', '1.0', 'Iris-versicolor'],
       ['6.6', '2.9', '4.6', '1.3', 'Iris-versicolor'],
       ['5.2', '2.7', '3.9', '1.4', 'Iris-versicolor'],
       ['5.0', '2.0', '3.5', '1.0', 'Iris-versicolor'],
       ['5.9', '3.0', '4.2', '1.5', 'Iris-versicolor'],
       ['6.0', '2.2', '4.0', '1.0', 'Iris-versicolor'],
       ['6.1', '2.9', '4.7', '1.4', 'Iris-versicolor'],
       ['5.6', '2.9', '3.6', '1.3', 'Iris-versicolor'],
       ['6.7', '3.1', '4.4', '1.4', 'Iris-versicolor'],
       ['5.6', '3.0', '4.5', '1.5', 'Iris-versicolor'],
       ['5.8', '2.7', '4.1', '1.0', 'Iris-versicolor'],
       ['6.2', '2.2', '4.5', '1.5', 'Iris-versicolor'],
       ['5.6', '2.5', '3.9', '1.1', 'Iris-versicolor'],
       ['5.9', '3.2', '4.8', '1.8', 'Iris-versicolor'],
       ['6.1', '2.8', '4.0', '1.3', 'Iris-versicolor'],
       ['6.3', '2.5', '4.9', '1.5', 'Iris-versicolor'],
       ['6.1', '2.8', '4.7', '1.2', 'Iris-versicolor'],
       ['6.4', '2.9', '4.3', '1.3', 'Iris-versicolor'],
       ['6.6', '3.0', '4.4', '1.4', 'Iris-versicolor'],
       ['6.8', '2.8', '4.8', '1.4', 'Iris-versicolor'],
       ['6.7', '3.0', '5.0', '1.7', 'Iris-versicolor'],
       ['6.0', '2.9', '4.5', '1.5', 'Iris-versicolor'],
       ['5.7', '2.6', '3.5', '1.0', 'Iris-versicolor'],
       ['5.5', '2.4', '3.8', '1.1', 'Iris-versicolor'],
       ['5.5', '2.4', '3.7', '1.0', 'Iris-versicolor'],
       ['5.8', '2.7', '3.9', '1.2', 'Iris-versicolor'],
       ['6.0', '2.7', '5.1', '1.6', 'Iris-versicolor'],
       ['5.4', '3.0', '4.5', '1.5', 'Iris-versicolor'],
       ['6.0', '3.4', '4.5', '1.6', 'Iris-versicolor'],
       ['6.7', '3.1', '4.7', '1.5', 'Iris-versicolor'],
       ['6.3', '2.3', '4.4', '1.3', 'Iris-versicolor'],
       ['5.6', '3.0', '4.1', '1.3', 'Iris-versicolor'],
       ['5.5', '2.5', '4.0', '1.3', 'Iris-versicolor'],
       ['5.5', '2.6', '4.4', '1.2', 'Iris-versicolor'],
       ['6.1', '3.0', '4.6', '1.4', 'Iris-versicolor'],
       ['5.8', '2.6', '4.0', '1.2', 'Iris-versicolor'],
       ['5.0', '2.3', '3.3', '1.0', 'Iris-versicolor'],
       ['5.6', '2.7', '4.2', '1.3', 'Iris-versicolor'],
       ['5.7', '3.0', '4.2', '1.2', 'Iris-versicolor'],
       ['5.7', '2.9', '4.2', '1.3', 'Iris-versicolor'],
       ['6.2', '2.9', '4.3', '1.3', 'Iris-versicolor'],
       ['5.1', '2.5', '3.0', '1.1', 'Iris-versicolor'],
       ['5.7', '2.8', '4.1', '1.3', 'Iris-versicolor']])


# In[198]:


for flower in data:
    if flower[4] == "Iris-setosa":
        flower[4] = 1
    else:
        flower[4]= 0

data = data.astype('float')    


# #### sklearn사용해서 랜덤하게 데이터 섞는다

# In[199]:


from sklearn.model_selection import train_test_split
X=data[:,:4]
y = data[:,4]
X_train, X_test, y_train, y_test = train_test_split(X,y)


# ### Main Function

# parameter: learning rate, hidden layer당 node 갯수, epochs (나중에 미니 배치 방식일때는 batch_size) 

# In[275]:


class IrisClassification(object):
    def __init__(self, n_x, n_h, n_y, eta = 0.1, epochs = 1, random_seed=1):
        self.n_x = n_x
        self.n_h = n_h
        self.n_y = n_y
        self.eta = eta
        self.epochs = epochs
        np.random.seed(random_seed)
        self.W1 = 2*np.random.random((self.n_x , self.n_h)) - 1  # between -1 and 1
        self.W2 = 2*np.random.random((self.n_h , self.n_y )) - 1  # between -1 and 1
        print('W1.shape={}, W2.shape={}'.format(self.W1.shape, self.W2.shape))
        
    def forpass(self, A0):
        Z1 = np.dot(A0, self.W1)          
        A1 = self.g(Z1)                   
        Z2 = np.dot(A1,self.W2)          
        A2 = self.g(Z2)     
        return Z1, A1, Z2, A2

    def fit(self, X, y): 
        self.cost_ = []
        self.m_samples = len(y)
        
        for epoch in range(self.epochs):
            print('Training epoch {}/{}'.format(epoch+1, self.epochs))
            
            for m in range(self.m_samples):            
                A0 = np.array(X[m], ndmin=2)    
                Y0 = np.array(y[m], ndmin=2)    

                Z1, A1, Z2, A2 = self.forpass(A0)          

                E2 = Y0 - A2                       
                E1 = np.dot(E2, self.W2.T)         

               
                dZ2 = E2 * self.g_prime(Z2)       
                dZ1 = E1 * self.g_prime(Z1)     

                
                self.W2 +=  self.eta * np.dot (A1.T, dZ2)     
                self.W1 +=  self.eta * np.dot( A0.T, dZ1)    
                
                self.cost_.append(np.sqrt(np.sum(E2 * E2)))
        return self

    def predict(self, X):
        A0 = np.array(X, ndmin=2)    
        Z1, A1, Z2, A2 = self.forpass(A0)   
        return A2                                       

    def g(self, x):                 # 시그모이드 함수
        result = 1.0/(1.0+np.exp(-x)) 
        #print("계산된 시그모이드 값은: "+str(np.round(result,2)))
        return result
                                    
    def g_prime(self, x):                    
        return self.g(x) * (1 - self.g(x))
    
    #잘 되었는지 test데이터로 확인해보자
    def evaluate(self, Xtest, ytest): 
        scores = 0        
        A2 = self.predict(Xtest)
        errors = ytest-A2
        return errors


# In[276]:


Iris =IrisClassification(4,10, 1, epochs = 4)
Iris.fit(X_train,y_train)


# ### Cost값

# In[273]:


Iris.cost_


# In[274]:


plt.plot(range(len(Iris.cost_)), Iris.cost_)
plt.xlabel('Epochs')
plt.ylabel('Error Squared Sum')
plt.title('MultiLayered Perceptron of Iris Data')
plt.show()


# #### 전체 에러값의 평균

# In[277]:


Iris.evaluate(X_test,y_test)

