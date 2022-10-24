import pandas as pd
import warnings
from sklearn.metrics import r2_score
warnings.filterwarnings('ignore')

temp=pd.read_excel('Q4.result.data.xlsx')
print(temp)
tempp=temp[(temp['放牧小区（plot）']=='G21' ) &(temp['月份']==9 )]
tempp.reset_index(inplace=True,drop=True)
tempp=tempp.groupby('年份').mean()
tempp.reset_index(inplace=True,drop=False)
print(tempp)  #5*17  5行 17列  都是9月的数据，2012 ，2014,2016,18,20的

import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential, load_model
np.set_printoptions(suppress=True)


from sklearn.preprocessing import MinMaxScaler
from pylab import *
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns
sns.set_palette("rainbow") #设置所有图的颜色，使用hls色彩空间
import numpy as np
from sklearn import metrics
from sklearn.metrics import mean_squared_error #均方误差
from sklearn.metrics import mean_absolute_error #平方绝对误差

x=tempp['土壤湿度'].values
y=tempp['有机物含量'].values
n=[2012,2014,2016,2018,2020]

print(x)
print(y)

size=x+y+12
print(size)

#[23.32212333  2.90493286  0.62820579 22.58110661 36.26184118]
size=[2330,290,60,2250,3620]
size=[1165,145,30,1125,1810]
# rng=np.random.RandomState()
# colors=rng.rand(100)
plt.scatter(x,y,s=size,alpha=0.7,c='green')  #,cmap="cividis"c=colors,alpha默认为1的！！！！
# plt.colorbar()

plt.axhline(y=y.mean(),ls="--")
plt.axvline(x=x.mean(),ls="--")
plt.legend()
plt.xlabel( '土壤湿度')
plt.ylabel( '有机物含量')
plt.title( '2012-2020 G21 9 月份 土地状态')
plt.savefig('./Q6 fig/scatter G21 9月份 土地状态.jpg')

fig,ax=plt.subplots()
ax.scatter(x,y,c='green',s=size,alpha=0.6,cmap="Wistia")

for i,txt in enumerate(n):
    ax.annotate(txt,(x[i],y[i]))

# plt.figure(dpi=300,figsize=(12,12))
plt.axhline(y=y.mean(),ls="--")
plt.axvline(x=x.mean(),ls="--")
plt.grid()
plt.legend()
plt.xlabel( '土壤湿度(kg/m2)')
plt.ylabel( '有机物含量')
plt.title( '2012-2020 9月份土地状态：土壤湿度、土壤肥力')
plt.savefig('./Q6 fig/2012~2022 G21 9月份 土地状态.jpg')

dataset=temp[['土壤湿度', '有机物含量']]
#########LSTM多变量模型#############
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        end_ix = i + n_steps
        if end_ix > len(sequences)-1:
            break
        # 最关键的不一样在这一步
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)
def mean_absolute_percentage_error(y_true, y_pred):

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def fitlstmmodel(dataset,n_steps=1):
    #dataset：数据标准化后的dataset
    # n_steps：分片大小，默认为1
    #依次为：'PM2.5','AQI',  'PM10', 'SO2', 'CO', 'NO2', 'O3_8h', '最高气温', '最低气温'

    in_seq1= dataset[:,0].reshape((dataset.shape[0], 1))
    in_seq2= dataset[:,1].reshape((dataset.shape[0], 1))

    dataset = np.hstack((in_seq1, in_seq2))
    X, y = split_sequences(dataset, n_steps)
    n_features = X.shape[2]
    model = Sequential()
    model.add(LSTM(300, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(LSTM(300, activation='relu'))

    # 和多对一不同点在于，这里多对多的Dense的神经元=features数目
    model.add(Dense(n_features))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=100, verbose=2,shuffle=False)
    model.save('lstm_model.h5')
    last_input=np.array(dataset[-1:,:])
    return X,y,last_input,n_features,n_steps

# 将整型变为float
dataset = dataset.astype('float32')
#对数据集合进行标准化
scaler = MinMaxScaler(feature_range=(0, 1))

dataset=scaler.fit_transform(dataset)
#输入为标准化后的dataset 	#输出：X为lstm的输入，y为lstm的输出，x_input_last为最后一行dataset的数据，用于预测未来的输入,n_features是特征维度，n_steps是切片分层
X,y,last_input,n_features,n_steps=fitlstmmodel(dataset,n_steps=1)
#输入1为lstm的输入X，输入2为lstm的输出y，用于训练模型,输入3为标准化模型
#输出：testPredict为预测close的训练数据，testY为close的真实数据
#该函数目标输出训练的RMSE以及预测与训练数据的对比

# 将整型变为float
dataset = dataset.astype('float32')
#对数据集合进行标准化
scaler = MinMaxScaler(feature_range=(0, 1))

dataset=scaler.fit_transform(dataset)
#输入为标准化后的dataset 	#输出：X为lstm的输入，y为lstm的输出，x_input_last为最后一行dataset的数据，用于预测未来的输入,n_features是特征维度，n_steps是切片分层
X,y,last_input,n_features,n_steps=fitlstmmodel(dataset,n_steps=1)
#输入1为lstm的输入X，输入2为lstm的输出y，用于训练模型,输入3为标准化模型
#输出：testPredict为预测close的训练数据，testY为close的真实数据
#该函数目标输出训练的RMSE以及预测与训练数据的对比


###预测与评分
def Predict_RMSE_BA(X, y, scaler):
    model = load_model('lstm_model.h5')
    trainPredict = model.predict(X)
    testPredict = scaler.inverse_transform(trainPredict)
    testY = scaler.inverse_transform(y)
    score(testY[:, 0], testPredict[:, 0])

    # 土壤湿度', '有机物含量'
    plt.plot(testY[:, 0], color='blue', label='observed data')
    plt.plot(testPredict[:, 0], color='red', label='LSTM')
    plt.xlabel('年份')
    plt.ylabel('土壤湿度')
    plt.title('2012~2020 9 月份 G21 土壤湿度')
    plt.legend()  # 显示图例

    plt.savefig('./Q6 fig/2012~2020 9 月份 G21 土壤湿度.jpg')
    plt.show()
    plt.close()

    score(testY[:, 1], testPredict[:, 1])
    plt.plot(testY[:, 1], color='blue', label='observed data')
    plt.plot(testPredict[:, 1], color='red', label='LSTM')
    plt.xlabel('年份')
    plt.ylabel('有机物含量')
    plt.title('2012~2022 9 月份 G21 有机物含量')
    plt.legend()  # 显示图例

    plt.savefig('./Q6 fig/2012~2022 9 月份 G21 有机物含量.jpg')
    plt.show()
    plt.close()

    return testPredict, testY


def score(y_true, y_pre):
    # MSE
    print("MAPE :")
    print(mean_absolute_percentage_error(y_true, y_pre))
    # RMSE
    print("RMSE :")
    print(np.sqrt(metrics.mean_squared_error(y_true, y_pre)))
    # MAE
    print("MAE :")
    print(metrics.mean_absolute_error(y_true, y_pre))
    # # R2
    # print("R2 :")
    # print(np.abs(r2_score(y_true,y_pre)))


testPredict, testY = Predict_RMSE_BA(X, y, scaler)


def Predict_future_plot(predict_forword_number, x_input, n_features, n_steps, scaler, testPredict, testY):
    model = load_model('lstm_model.h5')
    predict_list = []
    predict_list.append(x_input)
    while len(predict_list) < predict_forword_number:
        x_input = predict_list[-1].reshape((-1, n_steps, n_features))
        yhat = model.predict(x_input, verbose=0)
        # 预测新值
        predict_list.append(yhat)
    # 取出

    Predict_forword = scaler.inverse_transform(np.array([i.reshape(-1, 1)[:, 0].tolist() for i in predict_list]))
    return Predict_forword[1:, :].tolist()


y_pre = Predict_future_plot(2, last_input, n_features, n_steps, scaler, testPredict, testY)
print('y_pre',y_pre)

x=tempp['土壤湿度'].values
y=tempp['有机物含量'].values
x=np.append(x,31.243684784838464)
y=np.append(y,-6.185738463134904)
n=[2012,2014,2016,2018,2020,2022]

fig,ax=plt.subplots()
ax.scatter(x,y,c='r')

for i,txt in enumerate(n):
    ax.annotate(txt,(x[i],y[i]))

plt.axhline(y=y.mean(),ls="-")
plt.axvline(x=x.mean(),ls="-")
plt.legend()
plt.xlabel( '土壤湿度')
plt.ylabel( '有机物含量')
plt.title( './Q6 fig/2012-2022 G21 9 月份 土地状态')
plt.savefig('./Q6 fig/2012-2022 G21 9月份 土地状态.jpg')
