#!/usr/bin/env python
# coding: utf-8

# # Определение стоимости автомобилей

# Сервис по продаже автомобилей с пробегом «Не бит, не крашен» разрабатывает приложение для привлечения новых клиентов. В нём можно быстро узнать рыночную стоимость своего автомобиля. В вашем распоряжении исторические данные: технические характеристики, комплектации и цены автомобилей. Вам нужно построить модель для определения стоимости. 
# 
# Заказчику важны:
# 
# - качество предсказания;
# - скорость предсказания;
# - время обучения.

# ## Подготовка данных

# In[28]:


pip install lightgbm


# In[29]:


pip install catboost


# In[30]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import (StandardScaler,  OneHotEncoder)
from numpy.random import RandomState
from sklearn.model_selection import train_test_split

import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model
from sklearn.dummy import DummyRegressor


# In[31]:


df = pd.read_csv('/datasets/autos.csv')


# In[32]:


df.shape


# Вывел размер фрейма данных

# In[33]:


df.info


# Вывел информацию 

# In[34]:


df.head()


# Вывел первые пять строк таблицы

# In[35]:


print(f'В датафрейме {df.duplicated().sum()} дубликатов')


# Так как в датафрейме имеется почтовый индекс владельца анкеты (пользователя) можно считать что это дубликаты, а не анкеты со схожими данными, принимаю решение удалить дубликаты

# In[36]:


df = df.drop_duplicates().reset_index(drop = True)
print(f'В датафрейме {df.duplicated().sum()} дубликатов')


# In[37]:


df.isnull().sum()


# Имеются пропуски в столбцах VehicleType(тип автомобильного кузова), Gearbox(тип коробки передач), Model(модель автомобиля), 
# FuelType(тип топлива), Repaired(была машина в ремонте или нет).
# 
# Могу предположить что пропуски в столбце Repaired просто не заполнялись пользователями так как машины небыли в ремонте и их можно заменить на значение no.

# In[38]:


df['Repaired'] = df['Repaired'].fillna('no')
df['VehicleType'] = df['VehicleType'].fillna('not_specified')
df['FuelType'] = df['FuelType'].fillna('not_specified')


# Заменил пропущеные значения столбца Repaired на значения "no"
# 
# Заменил пропущеные значения столбца VehicleType на значения "not_specified" т.к. тип кузова мало влияет на цену автомобиля одной марки.

# Цена автомобиля зависит от модели автомобиля так что для лучшего обучения модели предлагаю удалить строки с пропущеными значениями этого столбца, и другие строки с пропущеными значениями

# In[39]:


df.groupby(df['Gearbox'].isnull()).mean()


# In[40]:


df['Gearbox'] = df['Gearbox'].fillna('manual')


# Заменил пропущеные значения столбца Repaired на значения "manual" т.к. механическая коробка более распространена на более ранних моделях.

# Цена автомобиля зависит от модели автомобиля так что для лучшего обучения модели предлагаю удалить строки с пропущеными значениями этого столбца.

# In[41]:


data = df.dropna(subset=['Model']).reset_index(drop=True)
data.isnull().sum()


# In[42]:


data = data.drop(['PostalCode', 
     'NumberOfPictures', 
     'DateCreated', 
     'LastSeen', 
     'RegistrationMonth', 
     'DateCrawled'], axis = 1
)


# In[43]:


columns = ['Power', 'Kilometer', 'Price']
for column in columns:
    data[column].plot(kind='box', title=column)
    plt.show()
    print()


# Есть аномалии в данных

# Самым мощным легковым автомобилем является Devel Sixteen (5000 л. с.), но эта модель ещё не поступила в серийное производство, до него самым мощьным легковым автомобилем являетс Dagger GT (2500 л. с), поэтому уберём все обьявления с мощьностью выше 2500 л.с. , и ниже 0.75 л.с. т.к. самым маломощным автомобилем в мире является Benz Patent Motorwagen.

# In[44]:


#for column in columns:
    #upper_lim = data[column].quantile(.95)
    #lower_lim = data[column].quantile(.05)
    #data = data[(data[column] < upper_lim) & (data[column] > lower_lim)]
    
data = data.loc[((df['Power']<= 2500) & (data['Power']>= 0.75))]


# In[45]:


min(data['Kilometer']), max(data['Kilometer'])


# Пробег автомобилей минимальное 5000 км, максимально 150000 км это нормальный пробег изменять не буду

# Судя по графику есть автомобили стоимостью в 0, решаю оставить значения выше нуля.

# In[46]:


data = data[data['Price']>0]


# In[47]:


for column in columns:
    data[column].plot(kind='box', title=column)
    plt.show()
    print()


# аномалий не стало

# In[48]:


#data['VehicleType'] = pd.get_dummies(data['VehicleType'], drop_first=True)
#data['Gearbox'] = pd.get_dummies(data['Gearbox'], drop_first=True)
#data['Model'] = pd.get_dummies(data['Model'], drop_first=True)
#data['FuelType'] = pd.get_dummies(data['FuelType'], drop_first=True)
#data['Brand'] = pd.get_dummies(data['Brand'], drop_first=True)
#data['Repaired'] = pd.get_dummies(data['Repaired'], drop_first=True)


# Преобразовал признаки

# In[49]:


features = data.drop(['Price'], axis = 1)
target = data['Price']
state = RandomState(12345)
columns = features.keys()


# Выделил признаки и целевой признак

# In[51]:


features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, 
    train_size=0.6,
    random_state=state
)
features_valid, features_test, target_valid, target_test = train_test_split(
    features_valid, target_valid, 
    train_size=0.7, 
    random_state=state
)


# разделил

# In[52]:


enc = OrdinalEncoder()

features_train['VehicleType'] = enc.fit_transform(features_train[['VehicleType']])
features_valid['VehicleType'] = enc.transform(features_valid[['VehicleType']])
features_test['VehicleType'] = enc.transform(features_test[['VehicleType']])

features_train['Gearbox'] = enc.fit_transform(features_train[['Gearbox']])
features_valid['Gearbox'] = enc.transform(features_valid[['Gearbox']])
features_test['Gearbox'] = enc.transform(features_test[['Gearbox']])

features_train['Model'] = enc.fit_transform(features_train[['Model']])
features_valid['Model'] = enc.transform(features_valid[['Model']])
features_test['Model'] = enc.transform(features_test[['Model']])

features_train['FuelType'] = enc.fit_transform(features_train[['FuelType']])
features_valid['FuelType'] = enc.transform(features_valid[['FuelType']])
features_test['FuelType'] = enc.transform(features_test[['FuelType']])

features_train['Brand'] = enc.fit_transform(features_train[['Brand']])
features_valid['Brand'] = enc.transform(features_valid[['Brand']])
features_test['Brand'] = enc.transform(features_test[['Brand']])

features_train['Repaired'] = enc.fit_transform(features_train[['Repaired']])
features_valid['Repaired'] = enc.transform(features_valid[['Repaired']])
features_test['Repaired'] = enc.transform(features_test[['Repaired']])


# In[53]:


scaler = StandardScaler()
scaler.fit(features_train)
features_train = scaler.transform(features_train)

features_valid = scaler.transform(features_valid)

features_test = scaler.transform(features_test)


# Произвёл стандартизацию признаков

# ## Обучение моделей

# In[54]:


get_ipython().run_cell_magic('time', '', "\ncat_model = CatBoostRegressor(\n    loss_function='Logloss', objective='RMSE', l2_leaf_reg=10, iterations=10, \n    learning_rate=0.6, custom_metric='RMSE', eval_metric='RMSE', depth=16, thread_count = -1, \n    logging_level='Verbose')")


# In[55]:


get_ipython().run_cell_magic('time', '', 'cat_model.fit(features_train, target_train, verbose=False, plot=False)')


# In[56]:


get_ipython().run_cell_magic('time', '', 'predicted_cat = cat_model.predict(features_valid)')


# In[57]:


print(f'RMSE модели: {mean_squared_error(target_valid, predicted_cat, squared=False)}')


# Модель CatBoostRegressor показала результат 1735 при время обучения 12.4 s, время предсказания 12.8 ms

# In[58]:


get_ipython().run_cell_magic('time', '', '\nlgb_model = lgb.LGBMRegressor(n_estimators=50, random_state=state)')


# In[59]:


get_ipython().run_cell_magic('time', '', 'lgb_model.fit(features_train, target_train)')


# In[60]:


get_ipython().run_cell_magic('time', '', 'predicted_lgb = lgb_model.predict(features_valid)')


# In[61]:


print(f'RMSE модели: {mean_squared_error(target_valid, predicted_lgb, squared=False)}')


# Модель LGBMRegressor показала результат 1798 при времени обучения 2min 56s и времени предсказания 388 ms

# In[62]:


get_ipython().run_cell_magic('time', '', "model_k = KNeighborsRegressor(n_neighbors=5, weights='distance')")


# In[63]:


get_ipython().run_cell_magic('time', '', 'model_k.fit(features_train, target_train)')


# In[64]:


get_ipython().run_cell_magic('time', '', 'predicted_model_k = model_k.predict(features_valid)')


# In[65]:


print(f'RMSE модели: {mean_squared_error(target_valid, predicted_model_k, squared=False)}')


# RMSE модели: 1877

# ## Анализ моделей

# In[66]:


data_1 = [
    {'Время обучения':'12.4 s' , 'Скорость предсказания':'12.8 ms', 'RMSE': 1735},
    {'Время обучения':'2min 56s' , 'Скорость предсказания':'388 ms', 'RMSE': 1798},
    {'Время обучения':'3.12 s' , 'Скорость предсказания':'18.4 s', 'RMSE': 2291}] 
 
 
dframe = pd.DataFrame(data_1, index =['cat_model', 'lgb_model', 'model_k'])

display(dframe)


# Лучше всего себя показала модель CatBoostRegressor, время обучения и предсказания у неё по ~ 12 s и лучше RMSE, её и протестируем на тестовой выборке, так же проверим модели на адекватность.

# In[67]:


get_ipython().run_cell_magic('time', '', "predicted_cat_test = cat_model.predict(features_test)\nrmse_cat_test = mean_squared_error(target_test, predicted_cat_test, squared=False)\n\nprint(f'RMSE модели: {rmse_cat_test}')")


# In[68]:


dummy_model = DummyRegressor(strategy="mean").fit(features_train, target_train)

predicted_dummy = dummy_model.predict(features_test)
rmse_dummy = mean_squared_error(target_test, predicted_dummy, squared=False)

print(f'RMSE модели: {rmse_dummy}')


# На валидационных данных лучше всего себя показала модель CatBoostRegressor с показателем RMSE 1714.
# 
# Так же модели прошли проверку на адекватность так как тестовая модель показала слишком большой показатель RMSE.
