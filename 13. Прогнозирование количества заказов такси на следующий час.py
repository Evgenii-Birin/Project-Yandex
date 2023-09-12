#!/usr/bin/env python
# coding: utf-8

# <h1>Содержание<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Финальный-комментарий" data-toc-modified-id="Финальный-комментарий-1"><span class="toc-item-num">1&nbsp;&nbsp;</span><span style="color: green">Финальный комментарий<span></span></span></a></span></li><li><span><a href="#Комментарий-ревьювера-2" data-toc-modified-id="Комментарий-ревьювера-2-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Комментарий ревьювера 2</a></span></li><li><span><a href="#Комментарий-ревьювера" data-toc-modified-id="Комментарий-ревьювера-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Комментарий ревьювера</a></span></li><li><span><a href="#Подготовка" data-toc-modified-id="Подготовка-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Подготовка</a></span></li><li><span><a href="#Анализ" data-toc-modified-id="Анализ-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Анализ</a></span></li><li><span><a href="#Обучение" data-toc-modified-id="Обучение-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Обучение</a></span><ul class="toc-item"><li><span><a href="#LinearRegression" data-toc-modified-id="LinearRegression-6.1"><span class="toc-item-num">6.1&nbsp;&nbsp;</span>LinearRegression</a></span></li><li><span><a href="#LGBMRegressor" data-toc-modified-id="LGBMRegressor-6.2"><span class="toc-item-num">6.2&nbsp;&nbsp;</span>LGBMRegressor</a></span></li><li><span><a href="#CatBoostRegressor" data-toc-modified-id="CatBoostRegressor-6.3"><span class="toc-item-num">6.3&nbsp;&nbsp;</span>CatBoostRegressor</a></span></li></ul></li><li><span><a href="#Тестирование" data-toc-modified-id="Тестирование-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Тестирование</a></span></li><li><span><a href="#Чек-лист-проверки" data-toc-modified-id="Чек-лист-проверки-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Чек-лист проверки</a></span></li><li><span><a href="#Общий-комментарий" data-toc-modified-id="Общий-комментарий-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Общий комментарий</a></span></li></ul></div>

# #  Прогнозирование заказов такси

# Компания «Чётенькое такси» собрала исторические данные о заказах такси в аэропортах. Чтобы привлекать больше водителей в период пиковой нагрузки, нужно спрогнозировать количество заказов такси на следующий час. Постройте модель для такого предсказания.
# 
# Значение метрики *RMSE* на тестовой выборке должно быть не больше 48.
# 
# Вам нужно:
# 
# 1. Загрузить данные и выполнить их ресемплирование по одному часу.
# 2. Проанализировать данные.
# 3. Обучить разные модели с различными гиперпараметрами. Сделать тестовую выборку размером 10% от исходных данных.
# 4. Проверить данные на тестовой выборке и сделать выводы.
# 
# 
# Данные лежат в файле `taxi.csv`. Количество заказов находится в столбце `num_orders` (от англ. *number of orders*, «число заказов»).

# ## Подготовка

# In[1]:


import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from statsmodels. tsa.stattools import adfuller

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
from matplotlib.pyplot import figure


# In[2]:


df = pd.read_csv('/datasets/taxi.csv')


# Загрузил данные

# In[3]:


df.info()


# Вывел информацию о фрейме данных

# In[4]:


df['datetime'] = pd.to_datetime(df['datetime'])


# Перевёл столбец "datetime" в формат datetime

# In[5]:


df = df.set_index('datetime')
df.head()


# Сделал столбец "datetime" индексом

# In[6]:


df = df.resample('1H').sum()
df.head()


# Выполнил ресемплирование индекса по одному часу.

# ## Анализ

# In[7]:


df.index.is_monotonic


# Проверил временную последовательность распределения. Последовательность соблюдена.

# In[8]:


df.plot(figsize=(18,9))
plt.show()


# Визуализировал временной ряд.

# In[9]:


decomposed = seasonal_decompose(df)

print('='*116)
decomposed.trend.plot(figsize=(18,9), fontsize=14)
plt.xlabel('Время заказов', fontweight='bold', fontsize=14)
plt.ylabel('Число заказов', fontweight='bold', fontsize=14)
plt.title('Тренд', fontweight='bold', fontsize=20)
plt.show()
print('='*116)
print()
decomposed.seasonal.plot(figsize=(18,9), fontsize=14)
plt.xlabel('Время заказов', fontweight='bold', fontsize=14)
plt.ylabel('Число заказов', fontweight='bold', fontsize=14)
plt.title('Сезонная составляющая', fontweight='bold', fontsize=20)
plt.show()
print('='*116)
print()
decomposed.resid.plot(figsize=(18,9), fontsize=14)
plt.xlabel('Время заказов', fontweight='bold', fontsize=14)
plt.ylabel('Число заказов', fontweight='bold', fontsize=14)
plt.title('Остаток декомпозиции', fontweight='bold', fontsize=20)
plt.show()
print('='*116)


# In[10]:


decomposed.seasonal[:'2018-03-04 00:00:00'].plot(figsize=(18,9), fontsize=14, ax=plt.gca())
plt.xlabel('Время заказов', fontweight='bold', fontsize=14)
plt.ylabel('Число заказов', fontweight='bold', fontsize=14)
plt.title('Тренд', fontweight='bold', fontsize=20)
plt.show()
print('='*116)
print()
decomposed.seasonal[:'2018-03-04 00:00:00'].plot(figsize=(18,9), fontsize=14)
plt.xlabel('Время заказов', fontweight='bold', fontsize=14)
plt.ylabel('Число заказов', fontweight='bold', fontsize=14)
plt.title('Сезонная составляющая', fontweight='bold', fontsize=20)
plt.show()
print('='*116)
print()
decomposed.resid[:'2018-03-04 00:00:00'].plot(figsize=(18,9), fontsize=14)
plt.xlabel('Время заказов', fontweight='bold', fontsize=14)
plt.ylabel('Число заказов', fontweight='bold', fontsize=14)
plt.title('Остаток декомпозиции', fontweight='bold', fontsize=20)
plt.show()
print('='*116)


# In[11]:


adfuller(df['num_orders'], regression='ctt')


# На графике тренда по всей выборке наблюдается возрастающий тренд. На части данных(4 суток) видно что примерно с 6 часов утра и до 0 часов количество заказов возрастает, а с 0 часов до 6 утра число заказо падает.
# 
# Поскольку p-значение равно 0.02 что меньше 0,05, мы можем отвергнуть нулевую гипотезу.
# 
# Это означает, что временной ряд является стационарным.
# 
# График сезонной составляющей и остатка при визуализации всех данных являются стационарнами, так как их распределение со временем не меняется и потому что у них не меняется стандартное отклонение.
# 
# График сезонной составляющей и тренда при визуализации части данных(четверо суток) являются стационарнами, так как их распределение со временем не меняется и потому что у них не меняется стандартное отклонение.

# In[12]:


def make_features(df, max_lag, rolling_mean_size):
    df['dayofweek'] = df.index.dayofweek
    df['day'] = df.index.day
    df['hour'] = df.index.hour
    
    for lag in range(1, max_lag + 1):
        df['lag_{}'.format(lag)] = df['num_orders'].shift(lag)

    df['rolling_mean'] = df['num_orders'].shift().rolling(rolling_mean_size).mean()


make_features(df, 48, 48)

df.head()


# Выделил временные признаки

# In[13]:


df.dropna(axis=0,inplace=True)


# удалил строки с пропущеными значениями

# ## Обучение

# In[14]:


train, test = train_test_split(df, shuffle=False, test_size=0.1)
train, valid = train_test_split(train, shuffle=False, test_size=0.2)


# разделил фрейм на обучающую и тестовую выборки

# In[15]:


features_train = train.drop('num_orders', axis=1)
target_train = train['num_orders']

features_valid = valid.drop('num_orders', axis=1)
target_valid = valid['num_orders']

features_test = test.drop('num_orders', axis=1)
target_test = test['num_orders']


# разделил на признаки и основной признак

# ### LinearRegression

# In[16]:


model = LinearRegression()

model = model.fit(features_train, target_train)
predicted_valid = model.predict(features_valid)

print('RMSE валидационной выборки:', mean_squared_error(target_valid, predicted_valid, squared=False))


# ###  LGBMRegressor

# In[17]:


model_lgb = LGBMRegressor()

model_lgb.fit(features_train, target_train)
predicted_lgb_valid = model_lgb.predict(features_valid)

print('RMSE валидационной выборки:', mean_squared_error(target_valid, predicted_lgb_valid, squared=False))


# ### CatBoostRegressor

# In[18]:


model_cbr = CatBoostRegressor(verbose=0, n_estimators=100)

model_cbr.fit(features_train, target_train)
predicted_cbr_valid = model_cbr.predict(features_valid)

print('RMSE валидационной выборки:', mean_squared_error(target_valid, predicted_cbr_valid, squared=False))


# ## Тестирование

# Лучшую метрику RMSE показала модель LinearRegression её и проверим на тестовой выборке и на адекватность.

# In[19]:


predicted_test = model.predict(features_test)

print('RMSE тестовой выборки:', mean_squared_error(target_test, predicted_test, squared=False))


# In[20]:


dummy_regr = DummyRegressor(strategy="mean")

dummy_regr.fit(features_train, target_train)
predicted_dummy_test = dummy_regr.predict(features_test)

print('RMSE тестовой выборки:', mean_squared_error(target_test, predicted_dummy_test, squared=False))


# Модель LinearRegression на тестовой выборке показала результат 42, что соответствует условиям поставленой задачи.
# 
# Так же модель прошла проверку на адекватность так как значение метрики RMSE тестовой выборки модели DummyRegressor на много выше чем у LinearRegression.

# In[21]:


predict = pd.DataFrame(data=predicted_test,index=target_test.index) 

plt.figure(figsize=(18, 9), dpi=80)
plt.plot(target_test, label='Факт')
plt.plot(predict, color='red', label='Прогноз')
plt.legend()
plt.xlabel('Время заказов', fontweight='bold', fontsize=14)
plt.ylabel('Число заказов', fontweight='bold', fontsize=14)
plt.title('график Прогноз-Факт', fontweight='bold', fontsize=20)
plt.show()

