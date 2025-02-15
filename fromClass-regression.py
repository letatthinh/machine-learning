'''

Author
Date


'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Function
def standRegres(xArr,yArr):
    xMat = np.mat(xArr); yMat = np.mat(yArr)
    xTx = xMat.T*xMat
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws


# Function 
def rssError(yArr,yHatArr): #yArr and yHatArr both need to be arrays
    return ((yArr-yHatArr)**2).sum()

# Function
def tssError(yArr,yHatArr): #yArr and yHatArr both need to be arrays
    return ((yArr-yHatArr.mean())**2).sum()

# Function
def Rsquare(yArr,yHatArr): #yArr and yHatArr both need to be arrays
    rss = rssError(yArr,yHatArr)
    tss = tssError(yArr,yHatArr)
    return 1 - rss / tss


penguins = pd.read_csv("https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv",delimiter=",")

# Choose two variables from the dataset
lmPenguins = penguins[['...','...']]

lmPenguins = lmPenguins.dropna(axis=0)

lmPenguins['X0'] = 1

# Shuffle dataframe using sample function
lmPenguins = lmPenguins.sample(frac=1)

ratio = 0.75

total_rows = lmPenguins.shape[0]
train_size = int(total_rows*ratio)

train = lmPenguins[0:train_size]
test = lmPenguins[train_size:]


X_train = train[['X0', '...']].to_numpy()
y_train = train[['...']].to_numpy()

X_test = test[['X0', '...']].to_numpy()
y_test = test[['...']].to_numpy()



beta = standRegres(X_train, y_train)

y_hat = X_train * beta


Rsquare(y_train[:,0], y_hat[:,0].flatten().A1)


y_test_hat = X_test * beta

Rsquare(y_test[:,0], y_test_hat[:,0].flatten().A1)


# Code to display a graph of the training data and the computed regression line
# You do not need to do anything with the following code.
# I provide it so you can see the line you computed
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X_train[:, 1], y_train[:, 0], s=2, c='red')
ax.plot(X_train[:, 1], y_hat[:,0])
plt.title("'Predictions' Using Training Data (i.e. Error In)")
plt.show()

