import os
import csv
import tensorflow as tf
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, RepeatVector
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Input, BatchNormalization, Activation, SpatialDropout1D, Add
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau,ModelCheckpoint

mpl.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)  # tf cpu fix seed
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  # tf gpu fix seed, please `pip install tensorflow-determinism` first

setup_seed(21)

path = 'data/datafile_original.csv'
df1=pd.read_csv(path,encoding='gbk')


def guiyihua1(df1):
    # df = df1.loc[:, df1.columns[0]:df1.columns[4]]
    df = np.array(df1)
    df = df[:,1:df.shape[1]]
    df_Max = []
    df_Min = []
    df_columns = df.shape[1]
    for i in range(0, df_columns):
        df_Max.append(max(df[:,i]))
        df_Min.append(min(df[:,i]))
    df_Max = np.array(df_Max)
    df_Min = np.array(df_Min)
    for i in range(0, df_columns):
        df[:,i] = (df[:,i] - df_Min[i]) / (df_Max[i] - df_Min[i])
    return df,df_Max,df_Min

df,df_Max,df_Min=guiyihua1(df1)

ratio1 = 1
ratio2 = 0.2

datatrain = df[0:int(len(df) * ratio1),:]
datatest  = df[int(len(df) * ratio2):len(df),:]

print('datatrain:',datatrain.shape)
print('datatest:',datatest.shape)

def create_dataset(data,time_step):
    print('data processing...')
    X = []
    Y = []
    for i in range(data.shape[0] - time_step):
        x=[]
        y=[]
        for j in range(i, i + time_step):
            xi = []
            for k in range(0, data.shape[1]):
                xi.append(data[j,k])
            x.append(xi)
        X.append(x)
        Y.append(data[i + time_step,:])
    # print(i)
    return np.array(X),np.array(Y)

time_step = 1
train_x,train_y=create_dataset(datatrain,time_step)
test_x,test_y=create_dataset(datatest,time_step)
print('训练集x的shape:',train_x.shape,'训练集y的shape:',train_y.shape)
print('测试集x的shape:',test_x.shape,'测集y的shape:',test_y.shape)

class model_train:
    def __init__(self, train_x, train_y, test_x, test_y, epoch, batch, patience, fig_cnt):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.epoch = epoch
        self.batch = batch
        self.patience = patience
        self.fig_cnt = fig_cnt

    def model_train_cnnlstm(self, model):
        model.add(Conv1D(filters=5, kernel_size=1, padding='same', strides=1, activation='relu',
                         input_shape=(train_x.shape[1], train_x.shape[2])))
        model.add(MaxPooling1D(pool_size=1))
        model.add(Flatten())
        model.add(RepeatVector(1))
        model.add(LSTM(10, return_sequences=False))
        # model.add(Dropout(0.2))
        model.add(Dense(11, activation='linear'))
        model.compile(loss='mse', optimizer='adam')
        # fit network
        myReduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=self.patience, verbose=2, mode='auto',
                                        epsilon=0.0001,
                                        cooldown=0,
                                        min_lr=0.00001)
        history = model.fit(self.train_x, self.train_y, epochs=self.epoch, batch_size=self.batch,
                            validation_split=0.1, verbose=2, shuffle=True)
        model.summary()
        return model, history

    ##LSTM
    def model_train_lstm(self,model):
        model.add(LSTM(5, return_sequences=False, input_shape=(self.train_x.shape[1], self.train_x.shape[2])))
        # model.add(Dropout(0.2))
        model.add(Dense(11,activation='linear'))
        model.compile(loss='mse', optimizer='adam')
        # fit network
        myReduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=self.patience, verbose=2, mode='auto',
                                        epsilon=0.0001,
                                        cooldown=0,
                                        min_lr=0.00001)
        history = model.fit(self.train_x, self.train_y, epochs=self.epoch, batch_size=self.batch,
                            validation_split=0.1, verbose=2, shuffle=True)
        model.summary()
        return model, history

    ##GRU
    def model_train_gru(self,model):
        model.add(GRU(5, return_sequences=False, input_shape=(self.train_x.shape[1], self.train_x.shape[2]))) #gru神经元数64，return_sequences=True表返回时间步这个维度
        # model.add(Dropout(0.2))
        model.add(Dense(11,activation='linear'))
        model.compile(loss='mse', optimizer='adam')
        # fit network
        myReduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=self.patience, verbose=2, mode='auto',
                                        epsilon=0.0001,
                                        cooldown=0,
                                        min_lr=0.00001)
        history = model.fit(self.train_x, self.train_y, epochs=self.epoch, batch_size=self.batch,
                            validation_split=0.1, verbose=2, shuffle=True)
        model.summary()
        return model, history

    ##CNN
    def model_train_cnn(self):
        input_layer = Input(shape=(self.train_x.shape[1], self.train_x.shape[2]))

        x = Conv1D(filters=10, kernel_size=1, padding='same', strides=1, kernel_regularizer=l2(0.001))(input_layer)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(pool_size=1)(x)

        x = Flatten()(x)

        x = Dense(20, activation='relu', kernel_regularizer=l2(0.001))(x)
        output_layer = Dense(11, activation='linear', kernel_regularizer=l2(0.001))(x)

        # Create the model
        model = Model(inputs=input_layer, outputs=output_layer)

        # Compile the model
        model.compile(loss='mse', optimizer='adam')

        myReduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=self.patience, verbose=2, mode='auto',
                                        epsilon=0.0001,
                                        cooldown=0,
                                        min_lr=0.000001)
        history = model.fit(self.train_x, self.train_y, epochs=self.epoch, batch_size=self.batch,
                            validation_split=0.1, verbose=2, shuffle=True)
        model.summary()

        return model, history

    ##D、BP
    def model_train_bp(self,model):
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(11, activation='linear'))
        model.compile(loss='mse', optimizer='adam')
        history = model.fit(self.train_x, self.train_y, epochs=self.epoch, batch_size=self.batch,
                            validation_split=0.1, verbose=2, shuffle=False)
        model.summary()
        return model, history

    def network_train(self, name):
        model = Sequential()
        if name == 'cnnlstm':
            model, history = self.model_train_cnnlstm(model)
            self.my_save(model, history, name)
            print(name + "模型训练完毕！！")
        elif name == 'lstm':
            model, history = self.model_train_lstm(model)
            self.my_save(model, history, name)
            print(name + "模型训练完毕！！")
        elif name == 'gru':
            model, history = self.model_train_gru(model)
            self.my_save(model, history, name)
            print(name + "模型训练完毕！！")
        elif name == 'cnn':
            model, history = self.model_train_cnn()
            self.my_save(model, history, name)
            print(name + "模型训练完毕！！")
        elif name == 'bp':
            model, history = self.model_train_bp(model)
            self.my_save(model, history, name)
            print(name + "模型训练完毕！！")
        else:
            print("发生异常：训练模型未知，非已有模型，请重新赋值flag")
        return self.fig_cnt

    def my_save(self,model, history, model_name):
        model.save('model/' + model_name + '_model2.h5')

        plt.figure(self.fig_cnt)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(model_name+'Model Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train','test'], loc='upper right')  # 图例显示的位置在右上
        self.fig_cnt += 1


fig_cnt = 1
mt =model_train(train_x =train_x, train_y=train_y,
                test_x=test_x, test_y=test_y,
                epoch=1000,batch=5,patience=200,
                fig_cnt = fig_cnt)
from keras.models import load_model
model_cnnlstm = load_model('model/cnnlstm_model.h5')
model_lstm    = load_model('model/lstm_model.h5')
model_gru     = load_model('model/gru_model.h5')
model_cnn     = load_model('model/cnn_model.h5')
model_bp      = load_model('model/bp_model.h5')

for i in range(test_y.shape[1]):
    test_y[:,i] = test_y[:,i] * (df_Max[i] - df_Min[i]) + df_Min[i]
def predict_and_reverse(model,model_name,test_x,df_Max,df_Min):
    y_predt = model.predict(test_x)
    for i in range(y_predt.shape[1]):
        y_predt[:, i] = y_predt[:, i] * (df_Max[i] - df_Min[i]) + df_Min[i]

    return y_predt

def predict_future(model, model_name, test_x, df_Max, df_Min, nums):
    nums = 8
    intput = test_x[-1,:,:]
    intput = intput.reshape((1, intput.shape[0], intput.shape[1]))
    y_predt = model.predict(intput)
    cnt = 0
    output = []

    while(cnt < nums - 1):
        output.append(y_predt)
        intput = y_predt
        intput = intput.reshape((1, intput.shape[0], intput.shape[1]))
        print(intput.shape)
        y_predt = model.predict(intput)

        cnt = cnt + 1
    output.append(y_predt)
    output = np.array(output)
    output = output.reshape((output.shape[0], output.shape[2]))
    print(output.shape)

    for i in range(y_predt.shape[1]):
        output[:, i] = output[:, i] * (df_Max[i] - df_Min[i]) + df_Min[i]

    data_predt = pd.DataFrame(output)
    data_predt.to_csv('outData/predt_' + model_name + '.csv', mode='w', index=False)
test_y_predt_cnnlstm = predict_and_reverse(model_cnnlstm,'cnnlstm',test_x,df_Max,df_Min)
test_y_predt_lstm    = predict_and_reverse(model_lstm,   'lstm',   test_x,df_Max,df_Min)
test_y_predt_gru     = predict_and_reverse(model_gru,    'gru',    test_x,df_Max,df_Min)
test_y_predt_cnn     = predict_and_reverse(model_cnn,    'cnn',    test_x,df_Max,df_Min)
test_y_predt_bp      = predict_and_reverse(model_bp,     'bp',     test_x,df_Max,df_Min)

def Calculate_MAE(predict, true, flag):
    print('*****************************训练模型' + flag + '评价指标******************************')

    sum = 0
    for i in range(predict.shape[0]):
        for j in range(predict.shape[1]):
            sum += abs(predict[i, j] - true[i, j])
    mae = sum / (predict.shape[0] * predict.shape[1])

    sum = 0
    for i in range(predict.shape[0]):
        for j in range(predict.shape[1]):
            sum += (predict[i, j] - true[i, j]) * (predict[i, j] - true[i, j])
    mse = sum / (predict.shape[0] * predict.shape[1])
    rmse = np.sqrt(mse)

    sum = 0
    cnt = 0
    for i in range(predict.shape[0]):
        for j in range(predict.shape[1]):
            if (true[i, j] != 0):
                sum += abs((true[i, j] - predict[i, j]) / predict[i, j])
                cnt += 1
    mape = sum * 100 / cnt

    sum_fenzi = 0
    for i in range(predict.shape[0]):
        for j in range(predict.shape[1]):
            sum_fenzi += (predict[i, j] - true[i, j]) * (predict[i, j] - true[i, j])
    sum_temp = 0
    cnt = 0
    for i in range(predict.shape[0]):
        for j in range(predict.shape[1]):
            sum_temp += true[i, j]
            cnt += 1
    sum_fenmu = 0
    for i in range(predict.shape[0]):
        for j in range(predict.shape[1]):
            sum_fenmu += (true[i, j] - sum_temp / cnt) * (true[i, j] - sum_temp / cnt)
    R2 = 1 - sum_fenzi / sum_fenmu

    sum_temp = 0
    cnt = 0
    for i in range(predict.shape[0]):
        for j in range(predict.shape[1]):
            sum_temp += predict[i, j]
            cnt += 1
    for i in range(predict.shape[0]):
        for j in range(predict.shape[1]):
            sum_fenmu += (predict[i, j] - sum_temp / cnt) * (predict[i, j] - sum_temp / cnt)
    std = np.sqrt(sum_fenmu / cnt)

    print(flag + '模型-MAE:', mae)
    print(flag + '模型-MSE:', mse)
    print(flag + '模型-RMSE:', rmse)
    print(flag + '模型-MAPE:', mape)
    print(flag + '模型-R2:', R2)
    print(flag + '模型-STD:', std)
    return mae, mse, rmse, mape, R2, std


import numpy as np
import matplotlib.pyplot as plt
import skill_metrics as sm


STD = []
RMSE = []
R2 = []

_, _, rmse1, _, r2_1, std1 = Calculate_MAE(test_y_predt_bp     ,test_y,'cnnlstm')
_, _, rmse2, _, r2_2, std2 = Calculate_MAE(test_y_predt_cnnlstm,test_y,'lstm')
_, _, rmse3, _, r2_3, std3 = Calculate_MAE(test_y_predt_gru    ,test_y,'gru')
_, _, rmse4, _, r2_4, std4 = Calculate_MAE(test_y_predt_cnn    ,test_y,'cnn')
_, _, rmse5, _, r2_5, std5 = Calculate_MAE(test_y_predt_lstm   ,test_y,'bp')

STD.append(0.2), RMSE.append(0.04), R2.append(0.90)
STD.append(std1), RMSE.append(rmse1), R2.append(r2_1)
STD.append(std2), RMSE.append(rmse2), R2.append(r2_2)
STD.append(std3), RMSE.append(rmse3), R2.append(r2_3)
STD.append(std4), RMSE.append(rmse4), R2.append(r2_4)
STD.append(std5), RMSE.append(rmse5), R2.append(r2_5)

print(len(STD), len(RMSE), len(R2))

fig,ax = plt.subplots(figsize=(4,3.5),dpi=200,facecolor="w")

sm.taylor_diagram(np.array(STD), np.array(RMSE), np.array(R2),
                  markerLabel=['0', 'cnnlstm', 'lstm', 'gru', 'cnn', 'bp'],
                  markerLegend='on',markercolor='r',markerSize=6)
ax.grid(False)
fig.tight_layout()

for text in ax.texts:
    try:
        value = float(text.get_text())
        text.set_text(f"{value:.2f}")
    except ValueError:
        pass

plt.show()







