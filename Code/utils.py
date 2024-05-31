import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

#Data expansion by adding noises
def train_augment(trainX, trainY):
  print('Using 3X Data Augmentation!')
  row = len(trainX[:,0])
  column = len(trainX[0,:])
  standard_deviation = np.std(trainX,0)
  add_01 = np.zeros([row,column])
  add_02 = np.zeros([row,column])
  add_03 = np.zeros([row,column])
  for id in range(0,row):
      add_01[id,:] = 0.01 * standard_deviation * np.random.randn(1, column)
      add_02[id,:] = 0.02 * standard_deviation * np.random.randn(1, column)
      #add_03[id,:] = 0.03 * standard_deviation * np.random.randn(1, column)
  augment_trainX = np.row_stack((trainX,trainX+add_01,trainX+add_02))#, trainX+add_03))
  augment_trainY = np.row_stack((trainY,trainY,trainY))#,trainY))
  return augment_trainX, augment_trainY

def yHatYPlot(Y_true, Y_predict, R2_test, title = '95 Percentile MPS',  sampleIndex=0):
    plt.scatter(Y_true, Y_predict, s=0.7)
    max_range = np.max([np.max(Y_true), np.max(Y_predict)])
    plt.legend(['R^2: '+ str(R2_test)[0:6]] ,loc='lower right')
    x_line = np.linspace(0,max_range,100)
    plt.plot(x_line, x_line, color='r')
    plt.title(title)
    plt.ylabel('Predicted MPS')
    plt.xlabel('KTH MPS')
    return plt

def yHatYPlot_MPS(Y_true, Y_predict, title = '4124 MPS',  sampleIndex=0):
    plt.scatter(Y_true, Y_predict, s=0.7)
    max_range = np.max([np.max(Y_true), np.max(Y_predict)])
    rmse_test = np.sqrt(mean_squared_error(Y_true, Y_predict))
    plt.legend(['RMSE: '+ str(rmse_test)[0:6]],loc='lower right')
    x_line = np.linspace(0,max_range,100)
    plt.plot(x_line, x_line, color='r')
    plt.title(title)
    plt.ylabel('Predicted MPS')
    plt.xlabel('KTH MPS')
    return plt
