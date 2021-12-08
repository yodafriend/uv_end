from django.shortcuts import render
from keras import models
from keras import layers
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup 
import requests
from pprint import pprint
from joblib import load
model2 = load('./savedModels/model.joblib') 

def build_model(request):
    if request.method == 'POST':
        avg = request.POST['avg']
        min = request.POST['min']
        max = request.POST['max']
        hum= request.POST['hum']
        kk=request.POST['cc']
        kk3=request.POST['uv']

        ypred = model2.predict([[avg, min, max, hum,kk,kk3]])
        ypredresult=0
        print(ypredresult)
        if ypred<=20:
            ypred="낮음단계입니다"
        else:
            ypred="보통단계입니다."

        return render(request, 'index.html', {'result' : ypred,'ypredresult':ypredresult})
    df = pd.read_csv('indata123.csv',encoding='cp949')
    UV_sensor = df['uv_sensor']//10
    Volts=df['volts']
    df = df[['Temp_avg','Temp_min','Temp_max', 'Humidity','cloud cover','uv index']]
    x = df[:]
    UV_sensor_train, UV_sensor_test, UV_sensor_Y_train, UV_sensor_Y_test = train_test_split(df, UV_sensor, test_size=0.2, shuffle=True, random_state=1234)
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',input_shape=(df.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])


    #UV_sensorModel = build_model()
    UV_sensor_history = model.fit(UV_sensor_train, UV_sensor_Y_train, epochs=300, batch_size=10, verbose=2)

    predict = model.predict(UV_sensor_test)
    mse=mean_squared_error(predict, UV_sensor_Y_test)
    mse2=(mean_squared_error(predict, UV_sensor_Y_test))**0.5
    X_new = np.array([[6.4,4.6,12.8,84,9,2]])
    predict = model.predict(X_new)
    predval=predict[0]*10
    predvaluv=predval*0.4*5.625/1000
    if predvaluv <=0.2:
        predictindex="0~2사이입니다."
    elif predvaluv <=0.4:
        predictindex="3~4사이입니다."
    else :
        predictindex="매우 높습니다."
    predvalmal=(predval,"로 uv계산식 으로 환산을 해보면",predvaluv,"로 나오며 인덱스는",predictindex,"입니다.")
    if predict<=20:
        predict="낮음(0~2)단계입니다"
    else:
        predict="보통(3~5)단계입니다."
    url= 'https://search.naver.com/search.naver?query=날씨'
    res=requests.get(url)
    res.raise_for_status()
    soup=BeautifulSoup(res.text,"lxml")
    summary=soup.find("p",attrs={"class":"summary"}).get_text() #어제랑비교
    curr_temp= soup.find("div",attrs={"class":"temperature_text"}).get_text() #현재온도
    sun = soup.find("li",attrs={"class":"item_today type_sun"}).get_text() #일몰
    mise = soup.find("li",attrs={"class":"item_today level1"}).get_text() #미세먼지
    uv = soup.find("li",attrs={"class":"item_today level2"}).get_text() #자외선
    #min_temp = soup.find("li",attrs={"class":"item_today type_sun"}).get_text()
    gang = soup.find("dt",attrs={"class":"term"}).get_text() #강수
    su = soup.find("dd",attrs={"class":"desc"}).get_text() #확률
    url='http://www.climate.go.kr/home/09_monitoring/uv/uv_main'
    res=requests.get(url)
    res.raise_for_status()
    soup=BeautifulSoup(res.text,"lxml")
    
    return render(request,'index.html',{"mse":mse,"mse2":mse2,"predict":predict,"summary":summary,"curr_temp":curr_temp,"sun":sun,
    "mise":mise,"uv":uv,"gang":gang,"su":su,"predval":predval,"predvalmal":predvalmal,"predvaluv":predvaluv,"predictindex":predictindex})
    

def visual(request):
    return render(request,'visual.html')
def out(request):
    df = pd.read_csv('data1234.csv',encoding='cp949')
    CUV = df['cumulative UV']
    MUV = df['Maximum UV']


    df = df[['Aerosol','Temp_avg','Temp_min','Temp_max', 'Humidity','sunshine time','Total daylight hours','cloud cover']]
    CUV_X_train, CUV_X_test, CUV_Y_train, CUY_Y_test = train_test_split(df, CUV, test_size=0.2, shuffle=True, random_state=7)
    MUV_X_train, MUV_X_test, MUV_Y_train, MUV_Y_test = train_test_split(df, MUV, test_size=0.2, shuffle=True, random_state=7)
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',input_shape=(df.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])


    CUV_history = model.fit(df, CUV, epochs=100, batch_size=10, verbose=0)
    MUV_history = model.fit(df, MUV, epochs=100, batch_size=10, verbose=0)

    CUV_hat = model.predict(CUV_X_test)
    MUV_hat = model.predict(MUV_X_test)
    X_new = np.array([[38.2,5,-4.2,12.7,60.2,11.3,9.0,3.0]]) 
    CUVpredict = model.predict(X_new)
    X_new = np.array([[38.2,5,-4.2,12.7,60.2,11.3,9.0,3.0]]) 
    MUVpredict = model.predict(X_new)
    CUVmse=mean_squared_error(CUV_hat, CUY_Y_test)
    MUVmse=mean_squared_error(MUV_hat, MUV_Y_test)
    print("누적 자외선량의 mse 오차: ",mean_squared_error(CUV_hat, CUY_Y_test))
    print("최대 자외선량의 mse 오차: ",mean_squared_error(MUV_hat, MUV_Y_test))

    return render(request,'index.html',{"CUVmse":CUVmse,"MUVmse":MUVmse,"cuvpredict":CUVpredict,"muvpredict":MUVpredict})