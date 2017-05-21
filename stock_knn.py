import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing,cross_validation,neighbors
from sklearn.preprocessing import Imputer
from datetime import date
import datetime
import matplotlib.pyplot as plt
from matplotlib import style

#讀取pickle
pickle_in=open('test2.pickle','rb')
df=pickle.load(pickle_in)
print(df.tail())

#Define
df = df[['date','open', 'high', 'low', 'settlement', 'volume', 'close', 'open_int', 'close_best_bid', 'close_best_ask']]
#前處理
df['close_adjusted'] = df['close']
df['close_adjusted'].fillna(df['settlement'], inplace=True)
df['close_best_bid'].fillna(df['settlement'], inplace=True)
df['close_best_ask'].fillna(df['settlement'], inplace=True)
#df.tail()
#先處理一些 NaN 的資料，因為台灣指數期貨在每個月的第三個禮拜三結算，所以每週三的 close price 都是 NaN
#均線Code
df['close_mvag5'] = df['close_adjusted'].rolling(window=5).mean()
df['close_mvag20'] = df['close_adjusted'].rolling(window=20).mean()
#df.tail()

# 計算 RSV 公式:100%  *（第 n 日的收盤價 - 過去 n 日中最低價）/（過去 n 日最高價 - 過去 n 日中最低價）
df['RSV'] = 100* (( df['close'] - df['low'].rolling(window=9).min() ) / (df['high'].rolling(window=9).max() - df['low'].rolling(window=9).min()))

df['RSV'].fillna(method='bfill', inplace=True)

data = {'K9':[17],'D9':[39]}

# 計算每一天的 KD
for i in range(1,len(df.index)):
    K9_value = (1/3) * df['RSV'][i] + (2/3) * data['K9'][i-1]
    data['K9'].append(K9_value)
    D9_value = (2/3) * data['D9'][i-1] + (1/3) * data['K9'][i]
    data['D9'].append(D9_value)


# 把 KD 放進 DataFrame
df_KD = pd.DataFrame(data)
df = pd.concat([df, df_KD], axis=1, join_axes=[df.index])
#df.tail()
#為了方便接下來 training，我們直接把 K9、 D9 和 5 日、20 日均線，都往後 shift 一天
df[['y_close_mvag5','y_close_mvag20','y_K9','y_D9']] = df[['close_mvag5','close_mvag20','K9','D9']].shift(1)

df.set_index('date', inplace=True)
#拿2016~2017的資料作訓練
df = df.loc[date(2016,2,1):date(2017,5,10)]

style.use('fivethirtyeight')#好像是畫圖的套件
ax1=plt.subplot2grid((2,1),(0,0))
ax2=plt.subplot2grid((2,1),(1,0),sharex=ax1)
df[['close','close_mvag5','close_mvag20']].plot(ax=ax1,linewidth=2,color=['k','r','b'])
df[['RSV','D9','K9']].plot(ax=ax2,linewidth=2,color=['r','g','b'])
plt.tight_layout()
plt.savefig('graph1.png',dip=10000)
plt.legend(loc='upper left')
plt.show()

labels = []

for i in range(len(df.index)):
    if df['open'][i] < df['settlement'][i]:
        labels.append(1)
    else:
        labels.append(0)

df['labels'] = pd.Series(labels, index= df.index)

X = np.array(df[['y_K9','y_D9','y_close_mvag5','y_close_mvag20']])
X = preprocessing.scale(X)
y = np.array(df['labels'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)
#產生KNN分類器
clf = neighbors.KNeighborsClassifier()
#訓練資料
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
#印出正確率
print ('accuracy', accuracy)

# 做多的賺賠 = 收盤最佳買價 - 開盤價
df['long'] = df['close_best_bid'] - df['open']

# 做空的賺賠 = 開盤價 - 收盤最佳賣價
df['short'] = df['open'] - df['close_best_ask']

def backtest(dataframe, startday, classifier):

    # 先取出要回測的資料 
    df_backtest = dataframe.loc[startday:]

    # 匯入特徵值
    X_backtest = np.array(df_backtest[['y_K9', 'y_D9', 'y_close_mvag5', 'y_close_mvag20']])
    X_backtest = preprocessing.scale(X_backtest)

    # 讓機器預測
    forecast = classifier.predict(X_backtest)

    # 假設初始金錢為 100000
    deposit = 100000
    deposits = []
    right_call = 0
    right_put = 0
    wrong_call = 0
    wrong_put = 0

    # 開始跑從 2016 到今天，每一天的交易情形
    for i in range(len(df_backtest.index)):

        if forecast[i] == 1:
            # 因為是假設買一口大台，一點的賺賠是 200 元
            deposit += (df_backtest['long'][i] * 200)
            #print ('%d, %s long, because forecast is: %d, you earn: %d' %(df_backtest['open'][i], df_backtest.index[i].isoformat() , forecast[i], df_backtest['long'][i]))
            if df_backtest['long'][i] > 0:
                right_call +=1
            else:
                wrong_call +=1
        elif forecast[i] == 0:
            deposit += (df_backtest['short'][i] * 200)
            #print ('%d, %s short, because forecast is: %d, you earn: %d' %(df_backtest['open'][i], df_backtest.index[i].isoformat() , forecast[i], df_backtest['short'][i]))
            if df_backtest['short'][i] > 0:
                right_put +=1
            else:
                wrong_put +=1
        deposits.append(deposit)
        #print (deposit)

    # 這邊基本上只是賺賠結果視覺化
    df_backtest['deposits'] = pd.Series(deposits, index= df_backtest.index)
    right_score = right_put + right_call
    wrong_score = wrong_put + wrong_call 
    call_accuracy = right_call / float(right_call + wrong_call)
    put_accuracy = right_put / float(right_put + wrong_put)
    accuracy = right_score/float(right_score + wrong_score)

    print ('your final deposit is %d' %(deposit))
    print ('right call & put:', right_call, right_put, 'wrong call & put:', wrong_call, wrong_put)
    print ('call_accuracy: %f , put_accuracy: %f , accuracy: %f' %(call_accuracy, put_accuracy, accuracy))


    ax1 = plt.subplot2grid((2,1),(0,0))
    ax2 = plt.subplot2grid((2,1),(1,0), sharex= ax1)
    df_backtest[['close', 'close_mvag5', 'close_mvag20']].plot(ax= ax1, linewidth=2, color=['k','r','g','b'])
    df_backtest['deposits'].plot(ax= ax2, linewidth=2, color='k')
    plt.tight_layout()
    plt.legend(loc='upper left')
    plt.savefig('graph2.png',dip=10000)
    plt.show()

backtest(df,date(2016,2,1),clf)