# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%
np.random.seed(42)
area = np.random.randint(500,3500,100)
price = area * 300 + np.random.randint(-50000,50000,100)
data = pd.DataFrame({'Area':area,'Price':price})
print(data.head())


# %%
X = data['Area']
y = data['Price']

#define model
w = 0.0
b = 0.0
lr = 1e-5
num_iteration = 1000



#define cost function
def compute_loss(X,y,w,b):
    loss = 1/2 * np.mean((w*X +b -y)**2)
    return loss

def comput_gradient(X,y,w,b):
    m = len(X)
    dw = 1/m * np.sum((w*X + b - y)*X)
    db = 1/m * np.sum(w*X + b - y)
    return dw, db

def gradient_descent(X,y,w,b,lr,num_iteration):
    for i in range(num_iteration):
        dw ,db = comput_gradient(X,y,w,b)
        w = w - lr * dw
        b = b - lr * db
        if i % 100 == 0:
            print(f'Iteration{i},loss = {compute_loss(X,y,w,b)}')
    

# %%
gradient_descent(X,y,w,b,lr,num_iteration)

y_hat = w*X + b

plt.scatter(X,y,color='blue',label='Actual Prices')
plt.plot(X,y_hat,color='red',label='Prediction Prices',linewidth=2)
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Linear Regression')
plt.legend()
plt.show()


