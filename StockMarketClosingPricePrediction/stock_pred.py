import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=20,10

from sklearn.preprocessing import MinMaxScaler
Scaler=MinMaxScaler(feature_range=(0,1))

df=pd.read_csv("NSE-TATA.csv")
df.head()

df["Date"]=pd.to_datetime(df.Date,format="%Y-%m-%d")
df.index=df['Date']

plt.figure(figsize=(16,8))
plt.plot(df["Close"],label='Close Price history')
 
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense

Data=df.sort_index(ascending=True,axis=0)
New_Dataset=pd.DataFrame(index=range(0,len(df)),columns=['Date','Close'])

for i in range(0,len(Data)):
    New_Dataset["Date"][i]=Data['Date'][i]
    New_Dataset["Close"][i]=Data["Close"][i]
    

New_Dataset.index=New_Dataset.Date
New_Dataset.drop("Date",axis=1,inplace=True)

Final_Dataset=New_Dataset.values

train_data=Final_Dataset[0:987,:]
valid_data=Final_Dataset[987:,:]

Scaler=MinMaxScaler(feature_range=(0,1))
Scaled_Data=Scaler.fit_transform(Final_Dataset)

x_train_data,y_train_data=[],[]

for i in range(60,len(train_data)):
    x_train_data.append(Scaled_Data[i-60:i,0])
    y_train_data.append(Scaled_Data[i,0])
    
x_train_data,y_train_data=np.array(x_train_data),np.array(y_train_data)

x_train_data=np.reshape(x_train_data,(x_train_data.shape[0],x_train_data.shape[1],1))

lstm_model=Sequential()
lstm_model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train_data.shape[1],1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))




lstm_model.compile(loss='mean_squared_error',optimizer='adam')
lstm_model.fit(x_train_data,y_train_data,epochs=1,batch_size=1,verbose=2)

inputs_data=New_Dataset[len(New_Dataset)-len(valid_data)-60:].values
inputs_data=inputs_data.reshape(-1,1)
inputs_data=Scaler.transform(inputs_data)


X_test=[]
for i in range(60,inputs_data.shape[0]):
    X_test.append(inputs_data[i-60:i,0])
X_test=np.array(X_test)

X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
closing_price=lstm_model.predict(X_test)
closing_price=Scaler.inverse_transform(closing_price)

lstm_model.save("saved_lstm_model.h5")

train_data=New_Dataset[:987]
valid_data=New_Dataset[987:]
valid_data['Predictions']=closing_price
plt.plot(train_data["Close"])
plt.plot(valid_data[['Close',"Predictions"]])