import numpy as np
import pandas as pd

mean = []
std = []

def eval_hypothesis_function(w, x):
    """Evaluate the hypothesis function"""
    return np.matmul(w.T, x.T)

def remap(x):
    # print(x)
    NSamples,Nfeatures = x.shape
    remap = np.hstack((np.ones((NSamples,1)),x)).T
    return remap

def compute_gradient_of_cost_function(x, y, w):
    """compute gradient of cost function"""

    NSamples,Nfeatures = x.shape
    
    hypothesis_function = eval_hypothesis_function(w, x)

    residual =  np.subtract(hypothesis_function.T, y)

    m = (residual.T*x.T)
    
    s = np.sum(m,axis=1)

    gradient_of_cost_function = (s/NSamples)
    
    gradient_of_cost_function = np.reshape(gradient_of_cost_function,(Nfeatures,1))

    return gradient_of_cost_function


def compute_L2_norm(gradient_of_cost_function):
    """compute L2-norm of gradient of cost function"""
    # return np.sqrt(np.sum(np.multiply(gradient_of_cost_function.T,gradient_of_cost_function)))
    # return np.sqrt(np.sum(gradient_of_cost_function**2))
    return np.linalg.norm(gradient_of_cost_function)

def scale_features(dataX,label,*arg):
    # print(label)
    if label == "training":
        meanT = dataX.mean()
        stdT = dataX.std()
        dataScaled = ((dataX - meanT ) / stdT)
        return dataScaled, meanT, stdT
    if label == "testing":
        # meanA = mean
        # stdA = std
        # print(dataX)
        # print(arg[0])
        # print(arg[1])
        dataScaled = ((dataX - arg[0] ) / arg[1])
        # print(dataScaled)
        return dataScaled

def predict_last_mile(x_testing,w):
    return np.matmul(w.T, x_testing.T)

def show_last(price):
    print("-"*28)
    print("Last Mile Cost")
    print("-"*28)
    for x,y in zip(price,price):
        print(x)

def load_predict(path_and_filename,mean,std):
    testing_data = pd.read_csv(path_and_filename)

    filas = len(testing_data)

    columnas = len(list(testing_data))

    dataX = pd.DataFrame.to_numpy(testing_data.iloc[:,0:columnas])

    print("-"*28)
    print("Training Data")
    print("-"*28)
    for x,y in zip(dataX,dataX):
        print(x)


    dataXscaled=[]
    # print(mean)
    # print(std)

    for featureX,meanX,stdX in zip (dataX.T,mean,std):
        # print(featureX,meanX,stdX)
        dataScaled= scale_features(featureX,"testing",meanX,stdX)
        # print(dataScaled)
        dataXscaled.append(dataScaled)

    dataXscaled = np.array(dataXscaled).T

    print("-"*28)
    print("Training Data Scaled")
    print("-"*28)
    for x,y in zip(dataXscaled,dataXscaled):
        print(x)

    dataXscaled = remap(dataXscaled)
    # print(dataXscaled.T)
    return dataXscaled.T



def load_data(path_and_filename):
    """ load data from comma-separated-value (CSV) file """

    # load training-data file
    training_data = pd.read_csv(path_and_filename)

    filas = len(training_data)
    columnas = len(list(training_data))

    dataX = pd.DataFrame.to_numpy(training_data.iloc[:,0:columnas-1])

    dataY = pd.DataFrame.to_numpy(training_data.iloc[:,-1]).reshape(filas,1)

    print("-"*28)
    print("Training Data")
    print("-"*28)
    for x,y in zip(dataX,dataY):
        print(x,y)

    #escalado

    dataXscaled=[]

    for featureX in dataX.T:
        dataScaled, meanX, stdX = scale_features(featureX,"training")
        dataXscaled.append(dataScaled)
        mean.append(meanX)
        std.append(stdX)

    dataXscaled = np.array(dataXscaled).T

    print("-"*28)
    print("Training Data Scaled")
    print("-"*28)
    for x,y in zip(dataXscaled,dataX):
        print(x)

    # print(dataXscaled)

    #se agregan unos
    # unos = np.ones(shape=filas)
    # dataXscaled = np.column_stack((dataXscaled,unos))
    dataXscaled = remap(dataXscaled) 
        # print("Training Data Scaled")
    # for x,y in zip(dataXscaled.T,dataX):
    #     print(x)   

    # print("Training Data Scaled Ones")
    # for x,y in zip(dataXscaled.T,dataX):
    #     print(x)
    # print(dataXscaled.T)


    return dataXscaled.T, dataY, mean, std, columnas

def show_w(w):
    print("-"*28)
    print("W parameter")
    print("-"*28)
    for x in range(len(w)):
        print("w",x,":",float(w[x]))
    return None


def gradient_descent(x_training, y_training, w, stopping_criteria, learning_rate):
    """ run the gradient descent algorith for optimisation"""

    # gradient descent algorithm
    L2_norm = 100.0
    i = 0
    while L2_norm > stopping_criteria:

        # compute gradient of cost function
        gradient_of_cost_function = compute_gradient_of_cost_function(x_training,y_training,w)
        # print(gradient_of_cost_function)

        # update parameters
        w = w - learning_rate*gradient_of_cost_function

        # compute L2 Norm
        # print(L2_norm)
        L2_norm = compute_L2_norm(gradient_of_cost_function)

        # print parameters
        # print('w:{}, L2:{}'.format(w, L2_norm))
        i += 1
    print(i)
    return w

