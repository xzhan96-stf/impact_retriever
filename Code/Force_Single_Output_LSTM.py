# MLP for Pima Indians Dataset Serialize to JSON and HDF5
#This model adds truncation
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers, regularizers, callbacks
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input
from tensorflow.keras.models import model_from_json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from utils import *  # Assuming you have a utils script
import os
import scipy.io as io
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import joblib
import warnings
import sys
warnings.filterwarnings('ignore')

# fix random seed for reproducibility
Dir_root = 'H:/Shared drives/CamLab Dataset_1/Helmet_Riddell/ML'
Dir_results = Dir_root + '/Results'
Dir_data_X = Dir_root + '/X/Signals'
Dir_data_Y = Dir_root + '/Y'
Dir_code = Dir_root + '/Code_LSTM'
np.random.seed(7)

def buildBaseModel(hidden_unit, dropout, output_nodes, lr, initialization, regularization, loss='mean_squared_error'):
    model = Sequential([Input(shape = (145, 48)),
                        LSTM(hidden_unit, return_sequences = True),
                        Dropout(rate = dropout),
                        LSTM(hidden_unit, return_sequences = False),
                        Dropout(rate = dropout),
                        Dense(output_nodes)
                        ])
    # Compile model
    Adam = optimizers.Adam(lr = lr, decay=5e-6)
    model.compile(loss=loss, optimizer=Adam, metrics = ['mean_absolute_error'])
    model.summary()
    return model

def modelFit(X_train, Y_train, X_val, Y_val, model, epoch, lr, batch_size=128, augment = False, verbose = False):
    if augment:
        X, Y = train_augment(X_train, Y_train)  # Standardized version
    else:
        X,Y = X_train, Y_train
    import time
    print('Start Training: ')
    lrreduce = callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=10, min_lr = 0.00005)
    tik = time.time()
    if verbose == True:
        history = model.fit(X, Y, validation_data=(X_val, Y_val), epochs=epoch, batch_size=batch_size, verbose=0, callbacks=[lrreduce])
    else:
        model.fit(X, Y, validation_data=(X_val, Y_val), epochs=epoch, batch_size=batch_size, verbose=0, callbacks=[lrreduce])
    tok = time.time()
  
    print('Training Time(s): ',(tok-tik))
    if verbose == True:
        plt.title("learning curve epoch: {}, lr: {}".format(str(epoch), str(lr)))
        loss, = plt.plot(history.history['loss'])
        val_loss, = plt.plot(history.history['val_loss'])
        plt.legend([loss, val_loss], ['loss', 'Val_loss'])
        plt.show()
        return model, (tok-tik), plt
    else:
        return model, (tok-tik)

def YPreprocessing(Y, method):
    if method == 'STD':
        Yscaler = StandardScaler()
        Yscaler.fit(Y)
        Y_out = Yscaler.transform(Y)
    elif method == 'LOG':
        Y_out = np.log(Y)
        Yscaler = None
    elif method == 'LOGSTD':
        Y_log = np.log(Y)
        Yscaler = StandardScaler()
        Yscaler.fit(Y_log)
        Y_out = Yscaler.transform(Y_log)
    else:
        Y_out = Y
        Yscaler = None
    return Y_out, Yscaler

def YTransform(Y, method, Yscaler=None):
    if method == 'STD':
        Y_out = Yscaler.transform(Y)
    elif method == 'LOG':
        Y_out = np.log(Y)
    elif method == 'LOGSTD':
        Y_log = np.log(Y)
        Y_out = Yscaler.transform(Y_log)
    else:
        Y_out = Y
    return Y_out

def YReconstruct(Y, method, Yscaler):
    if method == 'No':
        Y_out = Y
    elif method == 'LOG':
        Y_out = np.exp(Y)
    elif method == 'STD':
        Y_out = Yscaler.inverse_transform(Y)
    elif method == 'LOGSTD':
        Y_out = np.exp(Yscaler.inverse_transform(Y))
    return Y_out

def evaluate_truncated(y_test, y_pred, length):
    MAE = []
    RMSE = []
    for i in range(y_test.shape[0]):
        target = y_test[i,0:int(length[i])]
        output = y_pred[i,0:int(length[i])]
        MAE.append(mean_absolute_error(target, output))
        RMSE.append(np.sqrt(mean_squared_error(target, output)))
    MAE = np.mean(MAE)
    RMSE = np.mean(RMSE)
    R2 = r2_score(np.max(y_test,axis=1),np.max(y_pred,axis=1))
    return MAE,RMSE,R2

def evaluate(y_test, y_pred):
    MAE = mean_absolute_error(y_test, y_pred)
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
    R2 = r2_score(y_test,y_pred)
    return MAE,RMSE,R2

def main(outcome, hidden_unit, epoch):
    #Problem Definition
    task = 'Force'
    method = 'LSTM'
    dataset = 'mirror' #raw or augment
    augment = dataset == 'augment'
    #outcome_dict = {'F_Head_Face_X':0,'F_Head_Face_Y':1,'F_Head_Face_Z':2,'F_Head_Face_M':3,'F_Helmet_X':4,'F_Helmet_Y':5,'F_Helmet_Z':6,'F_Helmet_M':7,'F_Facemask_X':8,'F_Facemask_Y':9,'F_Facemask_Z':10,'F_Facemask_M':11}
    outcome_dict = {'Vel':0,'alpha':1,'beta':2,'x':3,'y':4,'z':5,'F_Head_Face':6,'F_Helmet':7,'F_Facemask':12,'F_Helmet_Facemask':13}
    Ymethod = 'No' #No
    feature_excluded = ''
    bootstrap = False
    print('Problem Definition: ' + task + ' ' + method + ' ' + Ymethod + ' ' + dataset + ' ' + outcome)

    #Load Dataset
    print('Loading Data!')
    os.chdir(Dir_data_X)
    X1_An = io.loadmat('Force_signal.mat')['signal_matrix']
    X2_An = io.loadmat('Force_mirror_signal.mat')['signal_matrix']
    X_An = np.concatenate([X1_An,X2_An],axis=2).transpose((2,1,0))
    
    X1_AnSPH = io.loadmat('Force_AnSPH_signal.mat')['signal_matrix']
    X2_AnSPH = io.loadmat('Force_mirror_AnSPH_signal.mat')['signal_matrix']
    X_AnSPH = np.concatenate([X1_AnSPH,X2_AnSPH],axis=2).transpose((2,1,0))
    
    X1_G = io.loadmat('Force_G_signal.mat')['signal_matrix']
    X2_G = io.loadmat('Force_mirror_G_signal.mat')['signal_matrix']
    X_G = np.concatenate([X1_G,X2_G],axis=2).transpose((2,1,0))
    
    X1_GSPH = io.loadmat('Force_GSPH_signal.mat')['signal_matrix']
    X2_GSPH = io.loadmat('Force_mirror_GSPH_signal.mat')['signal_matrix']
    X_GSPH = np.concatenate([X1_GSPH,X2_GSPH],axis=2).transpose((2,1,0))
    
    X = np.concatenate([X_An, X_AnSPH, X_G, X_GSPH], axis=1)
    
    assert X.shape[2] == 145
    assert X.shape[1] == 48
    assert X.shape[0] == 16000


    os.chdir(Dir_data_Y)
    ### --- Need Some Modification --- ###
    Y1 = io.loadmat('Force_Y.mat')['label'][:,[outcome_dict[outcome]]]
    Y2 = io.loadmat('Force_mirror_Y.mat')['label'][:,[outcome_dict[outcome]]]
    Y = np.concatenate([Y1, Y2], axis=0).reshape(-1,1,1)

    assert Y.shape[1] == 1
    assert Y.shape[0] == 16000
    
    
    X = np.transpose(X, [0,2,1])
    #Y = np.transpose(Y, [0,2,1])
    print('Entire Features: ', X.shape)
    print('Entire Labels: ', Y.shape)


    #Train/Val/Test Splits
    repeat = 0
    MAL_recorder = []
    MAE_recorder = []
    MAV_recorder = []
    # MAA_recorder = []
    RMSL_recorder = []
    RMSV_recorder = []
    RMSE_recorder = []
    # RMSA_recorder = []
    R2L_recorder = []
    R2V_recorder = []
    R2E_recorder =[]
    # R2A_recorder = []
    
    for repeat in range(17,20):
        random_seed = repeat*150
        X_train_val_id, X_test_id, Y_train_val_id, Y_test_id = train_test_split(np.arange(0,X.shape[0]), np.arange(0,Y.shape[0]), test_size=0.10, random_state=random_seed)
        X_train_id, X_val_id, Y_train_id, Y_val_id = train_test_split(X_train_val_id, Y_train_val_id, test_size=(0.10/0.90), random_state=random_seed)
        X_train, Y_train = X[X_train_id,:], Y[Y_train_id]
        X_train_val, Y_train_val = X[X_train_val_id,:], Y[Y_train_val_id]
        X_test, Y_test = X[X_test_id,:], Y[Y_test_id]
        X_val, Y_val = X[X_val_id,:], Y[Y_val_id]

        #Z-standardize the features
        X_train_std = X_train #Standardize the train/val set all together
        #Preprocessing the labels as intened (std, log, logstd)
        Y_train_std, Yscaler = YPreprocessing(Y=Y_train, method = Ymethod)
        print('Train Features: ', X_train_std.shape)
        print('Train Labels: ', Y_train_std.shape)
        print('Test Features: ', X_test.shape)
        print('Test Labels: ', Y_test.shape)
        print('Validation Features: ', X_val.shape)
        print('Validation Labels: ', Y_val.shape)
        
        #Model Definition Building and Parameterization
        input_nodes = X.shape[2]
        lr = 0.001
        output_nodes = 1
        dropout = 0.2
        regularization = 0.01
        initialization = "normal"
        loss = "mean_squared_error"
        model = buildBaseModel(hidden_unit, dropout, output_nodes, lr, initialization, regularization, loss='mean_squared_error')
        
        
        #Validation set performance test
        #Use training data to scale validation features and validation labels
        X_val_std = X_val
        Y_val_std = YTransform(Y=Y_val, method=Ymethod, Yscaler = Yscaler)
        model, train_time = modelFit(X_train_std, Y_train_std, X_val_std, Y_val_std, model, epoch = epoch, lr=lr, batch_size=512, augment = augment, verbose = False)

        print('Predicting Training Set and Validation Set!')
        Y_predict_train_raw = model.predict(X_train_std).squeeze()
        Y_predict_val_raw = model.predict(X_val_std).squeeze()
        #Reconstruct the Y_predicts based on the preprocessing methods
        print(Y_predict_train_raw.shape)
        print(Y_predict_val_raw.shape)
        Y_predict_train = YReconstruct(Y=Y_predict_train_raw, method = Ymethod, Yscaler = Yscaler)
        Y_predict_val = YReconstruct(Y=Y_predict_val_raw, method = Ymethod, Yscaler = Yscaler)

        MAL, RMSL, R2_train = evaluate(Y_train.squeeze(), Y_predict_train)
        MAV, RMSV, R2_val = evaluate(Y_val.squeeze(), Y_predict_val)
        print("MPS MAL: %.2f(%.2f)", MAL)
        print("MPS MAV: %.2f(%.2f)", MAV)
        print("MPS RMSL: %.2f(%.2f)", RMSL)
        print("MPS RMSV: %.2f(%.2f)", RMSV)
        print("MPS R2_train: %.2f(%.2f)", R2_train)
        print("MPS R2_val: %.2f(%.2f)", R2_val)

        #Test Set
        #Use the training+validation set to standardize the test features and labels
        X_train_val_std = X_train_val #Standardize the entire training set.
        X_test_std = X_test
        Y_train_val_std, Yscaler = YPreprocessing(Y = Y_train_val, method = Ymethod)
        model.fit(X_train_val_std, Y_train_val_std, epochs=5)

        import time
        tik = time.time()
        Y_predict_test_raw = model.predict(X_test_std).squeeze()
        tok = time.time()
        predict_time = tok - tik
        Y_predict_test = YReconstruct(Y = Y_predict_test_raw, method = Ymethod, Yscaler = Yscaler)

        MAE, RMSE, R2_test = evaluate(Y_test.squeeze(), Y_predict_test)
        print('Y_test shape: ', Y_test.shape)
        print('Y_predict shape: ',Y_predict_test.shape)
        print("MPS MAE: %.2f(%.2f)", MAE)
        print("MPS RMSE: %.2f(%.2f)", RMSE)
        print("MPS R2_test: %.2f(%.2f)", R2_test)
        print('MPS Predict Time: %.2f(%.2f)', predict_time)

        #Recording reproducible results
        MAL_recorder.append(MAL)
        MAE_recorder.append(MAE)
        MAV_recorder.append(MAV)
        RMSL_recorder.append(RMSL)
        RMSV_recorder.append(RMSV)
        RMSE_recorder.append(RMSE)
        R2L_recorder.append(R2_train)
        R2V_recorder.append(R2_val)
        R2E_recorder.append(R2_test)

        os.chdir(Dir_results+'/'+ task + '/' + method + '/' + dataset)
        specifics = 'Repeat' + str(repeat) + '_two-layer-LSTM_' + str(hidden_unit) + '_' + Ymethod + '_epoch' + str(epoch) 
        model_path = "."+"/"+ outcome + "/model/" + specifics + "_model.json"
        prediction = "."+"/"+ outcome + "/prediction/" + specifics

        io.savemat(prediction + 'id.mat',
                       {'X_train': X_train, 'Y_train': Y_train, 'X_val': X_val, 'Y_val': Y_val,
                        'X_test': X_test, 'Y_test': Y_test})
        io.savemat(prediction + '.mat',
                   {'Prediction': Y_predict_test, 'Prediction Val': Y_predict_val, 'Y_test': Y_test, 'Y_val': Y_val,
                    'Train Time': train_time,
                    'Prediction time': predict_time,
                    'RMSE': RMSE, 'MAL': MAL, 'MAV': MAV,
                    'MAE': MAE, 'RMSL': RMSL, 'RMSV': RMSV,
                    'R2_train': R2_train, 'R2_test': R2_test, 'R2_val': R2_val})

    # print(';'.join(['MAL']+[str(round(MAL_recorder[i],4)) for i in range(20)]))
    # print(';'.join(['MAV']+[str(round(MAV_recorder[i],4)) for i in range(20)]))
    # print(';'.join(['MAE']+[str(round(MAE_recorder[i],4)) for i in range(20)]))
    # #print('\t'.join(['MAA']+[str(round(MAA_recorder[i],4)) for i in range(20)]))
    # print(';'.join(['RMSL']+[str(round(RMSL_recorder[i],4)) for i in range(20)]))
    # print(';'.join(['RMSV']+[str(round(RMSV_recorder[i],4)) for i in range(20)]))
    # print(';'.join(['RMSE']+[str(round(RMSE_recorder[i],4)) for i in range(20)]))
    # #print('\t'.join(['RMSA']+[str(round(RMSA_recorder[i],4)) for i in range(20)]))
    # print(';'.join(['R2L']+[str(round(R2L_recorder[i],4)) for i in range(20)]))
    # print(';'.join(['R2V']+[str(round(R2V_recorder[i],4)) for i in range(20)]))
    # print(';'.join(['R2E']+[str(round(R2E_recorder[i],4)) for i in range(20)]))
    # #print('\t'.join(['R2A']+[str(round(R2A_recorder[i],4)) for i in range(20)]))



    group_name = 'YPreprocess'+ '_two-layer-LSTM_' + str(hidden_unit) + '_' + Ymethod + '_epoch' + str(epoch) 
    group_path = "."+"/"+ outcome + "/prediction/" + group_name
    # io.savemat(group_path + '.mat', 
    #            {'MAL': np.array(MAL_recorder).reshape(-1,1), 
    #             'MAV': np.array(MAV_recorder).reshape(-1,1), 
    #             'MAE': np.array(MAE_recorder).reshape(-1,1),
    #             'RMSL': np.array(RMSL_recorder).reshape(-1,1), 
    #             'RMSV': np.array(RMSV_recorder).reshape(-1,1), 
    #             'RMSE': np.array(RMSE_recorder).reshape(-1,1),
    #             'R2L': np.array(R2L_recorder).reshape(-1,1), 
    #             'R2V': np.array(R2V_recorder).reshape(-1,1), 
    #             'R2E': np.array(R2E_recorder).reshape(-1,1)})

    # serialize model to JSON

    model_json = model.to_json()
    with open(model_path, "w") as json_file:  #Save model skeleton
        json_file.write(model_json)
    # serialize weights to HDF5
    weight_path = model_path +"_weight.h5"
    model.save_weights(weight_path)  #Save model weight
    print("Saved model to disk!")

    # joblib_path = "."+"/"+ outcome + "/model/" + specifics + "_model.joblib"
    # joblib.dump(model,joblib_path)


    # with open(group_path + '.txt', 'w') as f:
    #     f.write('\t'.join(['MAL']+[str(round(MAL_recorder[i],4)) for i in range(20)]))
    #     f.write('\n')
    #     f.write('\t'.join(['MAV']+[str(round(MAV_recorder[i],4)) for i in range(20)]))
    #     f.write('\n')
    #     f.write('\t'.join(['MAE']+[str(round(MAE_recorder[i],4)) for i in range(20)]))
    #     f.write('\n')
    #     # f.write('\t'.join(['MAA']+[str(round(MAA_recorder[i],4)) for i in range(20)]))
    #     # f.write('\n')
    #     f.write('\t'.join(['RMSL']+[str(round(RMSL_recorder[i],4)) for i in range(20)]))
    #     f.write('\n')
    #     f.write('\t'.join(['RMSV']+[str(round(RMSV_recorder[i],4)) for i in range(20)]))
    #     f.write('\n')
    #     f.write('\t'.join(['RMSE']+[str(round(RMSE_recorder[i],4)) for i in range(20)]))
    #     f.write('\n')
    #     # f.write('\t'.join(['RMSA']+[str(round(RMSA_recorder[i],4)) for i in range(20)]))
    #     # f.write('\n')
    #     f.write('\t'.join(['R2L']+[str(round(R2L_recorder[i],4)) for i in range(20)]))
    #     f.write('\n')
    #     f.write('\t'.join(['R2V']+[str(round(R2V_recorder[i],4)) for i in range(20)]))
    #     f.write('\n')
    #     f.write('\t'.join(['R2E']+[str(round(R2E_recorder[i],4)) for i in range(20)]))
    #     # f.write('\n')
    #     # f.write('\t'.join(['R2A']+[str(round(R2A_recorder[i],4)) for i in range(20)]))

if __name__ == "__main__":
        #arguments receiver
    args = sys.argv[1:]
    outcome = str(args[0]) # Outcome to be predicted
    hidden_unit = int(args[1]) # Feature type to be used: G, An, GAn, GSPH, AnSPH, GGSPH, AnAnSPH, GGSPHAnAnSPH
    epoch = int(args[2]) # Number of epochs to train the model: 300, 500, 1000, 1500, 2000
    main(outcome = outcome, hidden_unit = hidden_unit, epoch = epoch)