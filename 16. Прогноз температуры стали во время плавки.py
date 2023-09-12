#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#План-работы" data-toc-modified-id="План-работы-1"><span class="toc-item-num">1&nbsp;&nbsp;</span><strong>План работы</strong></a></span></li><li><span><a href="#" data-toc-modified-id="-2"><span class="toc-item-num">2&nbsp;&nbsp;</span></a></span></li><li><span><a href="#" data-toc-modified-id="-3"><span class="toc-item-num">3&nbsp;&nbsp;</span></a></span></li><li><span><a href="#" data-toc-modified-id="-4"><span class="toc-item-num">4&nbsp;&nbsp;</span></a></span></li></ul></div>

#  **Промышленность — задача проекта**

# **Чтобы оптимизировать производственные расходы, металлургический комбинат ООО «Так закаляем сталь» решил уменьшить потребление электроэнергии на этапе обработки стали.**
# **Нам предстоит построить модель, которая предскажет температуру стали.**

# **Описание этапа обработки**
# 
# Сталь обрабатывают в металлическом ковше вместимостью около 100 тонн. Чтобы ковш выдерживал высокие температуры, изнутри его облицовывают огнеупорным кирпичом. Расплавленную сталь заливают в ковш и подогревают до нужной температуры графитовыми электродами. Они установлены в крышке ковша. 
# 
# Из сплава выводится сера (десульфурация), добавлением примесей корректируется химический состав и отбираются пробы. Сталь легируют — изменяют её состав — подавая куски сплава из бункера для сыпучих материалов или проволоку через специальный трайб-аппарат (англ. tribe, «масса»).
# 
# Перед тем как первый раз ввести легирующие добавки, измеряют температуру стали и производят её химический анализ. Потом температуру на несколько минут повышают, добавляют легирующие материалы и продувают сплав инертным газом. Затем его перемешивают и снова проводят измерения. Такой цикл повторяется до достижения целевого химического состава и оптимальной температуры плавки.
# 
# Тогда расплавленная сталь отправляется на доводку металла или поступает в машину непрерывной разливки. Оттуда готовый продукт выходит в виде заготовок-слябов (англ. *slab*, «плита»).

# **Описание данных**
# 
# Данные состоят из файлов, полученных из разных источников:
# 
# - `data_arc_new.csv` — данные об электродах;
# - `data_bulk_new.csv` — данные о подаче сыпучих материалов (объём);
# - `data_bulk_time_new.csv` *—* данные о подаче сыпучих материалов (время);
# - `data_gas_new.csv` — данные о продувке сплава газом;
# - `data_temp_new.csv` — результаты измерения температуры;
# - `data_wire_new.csv` — данные о проволочных материалах (объём);
# - `data_wire_time_new.csv` — данные о проволочных материалах (время).
# 
# Во всех файлах столбец `key` содержит номер партии. В файлах может быть несколько строк с одинаковым значением `key`: они соответствуют разным итерациям обработки.

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import probplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_val_score
from tensorflow.keras.models import Sequential
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Dense, Dropout, Flatten
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
import numpy as np


# In[2]:


try:    
    df_arc = pd.read_csv('D:\питон\DataFram\data_arc_new.csv')
    df_bulk = pd.read_csv('D:\питон\DataFram\data_bulk_new.csv')
    df_bulk_time = pd.read_csv('D:\питон\DataFram\data_bulk_time_new.csv')
    df_gas = pd.read_csv('D:\питон\DataFram\data_gas_new.csv')
    df_temp = pd.read_csv('D:\питон\DataFram\data_temp_new.csv')
    df_wire = pd.read_csv('D:\питон\DataFram\data_wire_new.csv')
    df_wire_time = pd.read_csv('D:\питон\DataFram\data_wire_time_new.csv')
except:    
    df_arc = pd.read_csv('/datasets/data_arc_new.csv')
    df_bulk = pd.read_csv('/datasets/data_bulk_new.csv')
    df_bulk_time = pd.read_csv('/datasets/data_bulk_time_new.csv')
    df_gas = pd.read_csv('/datasets/data_gas_new.csv')
    df_temp = pd.read_csv('/datasets/data_temp_new.csv')
    df_wire = pd.read_csv('/datasets/data_wire_new.csv')
    df_wire_time = pd.read_csv('/datasets/data_wire_time_new.csv')


# In[3]:


#name = {0:df_arc, 1:df_bulk, 2:df_bulk_time, 3:df_gas, 4:df_temp, 5:df_wire, 6:df_wire_time}

#for i in name.values():
    #display('-'*60)
    #display(i.info())
    #display('-'*60)
    


# In[4]:


#for i in name.values():
    #display('-'*80)
    #display(i.head(5))
    #display('='*80)


# **Таблица arc**

# In[5]:


df_arc.info()


# In[6]:


df_arc.head()


# In[7]:


df_arc.isnull().sum()


# In[8]:


df_arc.describe()


# In[9]:


sns.boxplot(data=df_arc)


# In[10]:


sns.boxplot(data=df_arc.drop(columns='key', axis=1))


# Есть выбросы в столбце 'Реактивная мощность' в отрицательную сторону тоесть значения ниже 0. Считаю что в дальнейшем их нужно будет удалить.

# In[11]:


df_arc.isnull().sum()


# Пропусков в таблице нет

# **Гистограммы**

# In[12]:


df_arc.hist(figsize=(10,10))
plt.show()


# По данной гистограмме не вполне ясно нормальное ли распределение у столбцов.

# **Графики Q-Q**

# In[13]:


probplot(x=df_arc['Активная мощность'],dist='norm',plot=plt)
plt.title('Активная мощность')
plt.show()


# In[14]:


probplot(x=df_arc['Реактивная мощность'],dist='norm',plot=plt)
plt.title('Реактивная мощность')
plt.show()


# Как видно из приведенных выше графиков Q-Q, столбец 'Реактивная мощность' близко следуют за красной линией (нормальное / гауссовское распределение). В то время как столбец 'Активная мощность' сильно удален от красной линии в нескольких местах, что уводит их далеко от гауссовости. Графики Q-Q более надежны, чем гистограммы.
# Так же видим аномалии с отрицательным значением.

# **Таблица bulk**

# In[15]:


display(df_bulk.info())
display(df_bulk.head())


# In[16]:


df_bulk.isnull().sum()


# В таблице bulk очень много пропусков скорее всего пропуски показывают что на данном этапе при произвотстве этой партии сыпучие материалы не добавлялись.
# В дальнейшем предлогаю заменить значения NAN на 0 что бы не возникало проблем с обучением модели

# In[17]:


df_bulk.describe()


# In[18]:


for i in df_bulk.drop(columns='key', axis=1).columns:
    sns.boxplot(data=df_bulk[i])
    plt.title(i)
    plt.show()


# Так же есть аномалии хорошо бы узнать у заказчика реальны такие обьёмы сыпучих материалов или ошибка в данных и уже после этого принимать решение удалять аномалии или оставить

# df_bulk.drop(columns='key', axis=1).hist(figsize=(10,10))
# plt.show()

# In[19]:


probplot(x=df_bulk['Bulk 11'],dist='norm',plot=plt)


# На некоторых гистограммах показывает нормальное распределение проверю на графиках Q-Q

# In[20]:


for i in df_bulk.drop(columns='key', axis=1).columns:
    probplot(x=df_bulk[i],dist='norm',plot=plt)
    plt.title(i)
    plt.show()


# Графики Q-Q не показывают нормального распределения.

# **Таблица bulk_time**

# In[21]:


df_bulk_time.info()


# In[22]:


df_bulk_time.head()


# In[23]:


df_bulk_time.isnull().sum()


# Так же много пропусков скорее всего они связаны с тем что на данном этапе не производилась загрузка сыпучих материалов и данные о времени не вносились. Предлогаю так же как и с прошлой таблицей заменить их на 0

# **Таблица gas**

# In[24]:


df_gas.info()


# In[25]:


df_gas.head()


# In[26]:


df_gas.isnull().sum()


# Пропусков в данных нет

# In[27]:


df_gas.describe()


# In[28]:


sns.boxplot(df_gas['Газ 1'])
plt.title('Газ 1')
plt.show()


# In[29]:


df_gas['Газ 1'].hist(figsize=(10,10))
plt.show()


# In[30]:


probplot(x=df_gas['Газ 1'], dist='norm', plot=plt)
plt.title('Газ 1')
plt.show()


# Наблюдаются аномалии, так же нужно узнать у заказчика подают ли такие обьёмы газа в ковш или ошибка в данных и на основе ответа принимать решение об удалении или оставлении этих значений.
# Так же по графикам заметно ненормальное распределение.

# **Таблица temp**

# In[31]:


df_temp.info()


# In[32]:


df_temp.head()


# In[33]:


df_temp.isnull().sum()


# Есть пропуски значений в столбце температура, предлагаю подставить температуру с ближайших значений той же партии так как удалив эти строки мы потеряем примерно 1/6 часть данных, а заменив их на среднее значение можем повлиять на точность обучения модели

# In[34]:


df_temp.describe()


# In[35]:


sns.boxplot(df_temp['Температура'])
plt.title('Температура')
plt.show()


# Есть аномалии я бы предложил удалить все вбросы показаний температур меньше 1500 градусов

# In[36]:


df_temp['Температура'].hist(figsize=(10,10))
plt.title('Температура')
plt.show()


# In[37]:


probplot(x=df_temp['Температура'], dist='norm', plot=plt)
plt.title('Температура')
plt.show()


# По графикам можно предположить нормальное распределение

# **Таблица wire**

# In[38]:


df_wire.info()


# In[39]:


df_wire.head()


# In[40]:


df_wire.isnull().sum()


# Очень много пропуско связаных скорее всего с тем что на данном этапе проволока не подавалась

# In[41]:


df_wire.describe()


# In[42]:


for i in df_wire.drop(columns='key', axis=1).columns:
    sns.boxplot(data=df_wire[i])
    plt.title(i)
    plt.show()


# Есть аномалии так же желательно узнать подаются ли такие обьемы проволоки

# In[43]:


df_wire.drop(columns='key', axis=1).hist(figsize=(10,10))
plt.show()


# In[44]:


for i in df_wire.drop(columns='key', axis=1).columns:
    probplot(x=df_wire[i],dist='norm',plot=plt)
    plt.title(i)
    plt.show()


# нормальное распределение можно предположить только у столбца 'Wire 1'

# **Таблица wire_time**

# In[45]:


df_wire_time.info()


# In[46]:


df_wire_time.head()


# In[47]:


df_wire_time.isnull().sum()


# Очень много пропущеных значений связаных скорее всего что на данном этапе в данной партии проволока не подовалась предлагаю так же заменить отсутствуещие значения на 0

# Качество данных номальное только надо разабратся с пропусками и перевести значения в цифровой вид

# **План работы**

# - произвести обработку данных
# - соеденить таблицы в одну методом merge по столбцу 'key'
# - разделить данные на обучающую валидационную и тестовую выборки
# - произвести разделение на признаки и целевой признак
# - произвети обучение нескольких моделей регрессии
# - выбрать модель с лучшими показателями
# - проверить её на тестовой выборке и на адекватность

# нужно узнать про аномальные значения и по какой метрике заказчик хотел бы видеть результат и в каком виде

# **Предобработка данных**

# **Таблица arc**

# In[48]:


df_arc[df_arc['Реактивная мощность'] < 0]


# In[49]:


df_arc[df_arc['key'] == 2116]


# Отрицательное значение находится не в последний строке партии значит удаляем только строку с вбросом

# In[50]:


df_arc = df_arc[df_arc['Реактивная мощность'] > 0]


# In[51]:


df_arc['Начало нагрева дугой'] = pd.to_datetime(df_arc['Начало нагрева дугой'])
df_arc['Конец нагрева дугой'] = pd.to_datetime(df_arc['Конец нагрева дугой'])


# Перевёл столбзы в формат даты и времени

# In[52]:


df_arc['Время нагрева дугой'] = df_arc['Конец нагрева дугой'] - df_arc['Начало нагрева дугой']


# In[53]:


df_arc.head(10)


# In[54]:


df_arc['Время нагрева дугой'] = df_arc['Время нагрева дугой'].apply(lambda x:x.seconds)
df_arc.head(10)


# Добавил столбец времени нагрева дугой

# In[55]:


df_arc['Полная мощность'] = (df_arc['Активная мощность']**2 + df_arc['Реактивная мощность']**2)**0.5
df_arc.head(10)


# Добавил столбец полная мощность

# In[56]:


df_arc['Потреблённая мощность'] = df_arc['Полная мощность'] * df_arc['Время нагрева дугой']
df_arc.head(10)


# Добавил столбец потреблённая мощьность

# In[57]:


df_arc_new = df_arc.drop(['Начало нагрева дугой', 'Конец нагрева дугой'], axis=1)
df_arc_new.head()


# In[58]:


counting = df_arc_new.groupby('key').count().reset_index()[['key','Время нагрева дугой']]
#counting = counting.rename(columns={'Время нагрева дугой':'Кол-во запусков'})
print(counting)


# In[59]:


df_arc_new = df_arc_new.groupby('key').sum()
df_arc_new.head(10)


# Удалил столбцы с датами и произвёл агрегацию столбцов с сумированием данных.

# In[60]:


df_arc_new['Кол-во запусков'] = counting['Время нагрева дугой']


# In[61]:


df_arc_new.isnull().sum()
df_arc_new.info()


# In[62]:


df_arc_new = df_arc_new.fillna(0)
df_arc_new = df_arc_new[df_arc_new['Кол-во запусков'] > 0]


# **Таблица bulk**

# In[63]:


df_bulk = df_bulk.fillna(0)
df_bulk.isnull().sum()


# Заменил пропущенные значения на 0.

# In[64]:


df_bulk['Sum Bulk'] = df_bulk.sum(axis=1)
df_bulk.head(10)


# In[65]:


df_bulk_new = df_bulk.drop(
    ['Bulk 1', 'Bulk 2', 'Bulk 3', 'Bulk 4',
     'Bulk 5', 'Bulk 6', 'Bulk 7', 'Bulk 8', 'Bulk 9',
     'Bulk 10', 'Bulk 11', 'Bulk 12', 'Bulk 13', 'Bulk 14', 'Bulk 15'], axis=1
)
df_bulk_new.head(10)


# Сделал таблицу с сумарным обьёмом сыпучих материалов

# **Таблица temp**

# In[66]:


df_temp.head(20)


# In[67]:


df_temp['Температура'].isnull()


# In[68]:


df_temp.info()


# In[69]:


df_temp[df_temp['Температура'].isnull() == True]


# Как мы видим если в измерении температуры ковша пропуск в последним измерении то и в остальных замерах этого ковша тоже пропуски

# In[70]:


x = df_temp[df_temp['Температура'].isnull() == True]
y = x['key'].unique()
print(y)


# Номера ковшей которые нужно удалить

# In[71]:


key_uni = df_temp['key'].unique()
temp_fin = []
for key in key_uni:
    coun = len(df_temp[df_temp['key']==key]['key'])
    temp_fin.append(
        [key, df_temp[df_temp['key']==key]['Температура'].iloc[0], df_temp[df_temp['key']==key]['Температура'].iloc[coun-1]]
    )
temp_fin = pd.DataFrame(data=temp_fin, columns=['key', 'Первая температура', 'Последняя температура'])
temp_fin.head()


# In[72]:


temp_fin = temp_fin[temp_fin['Первая температура'] > 1450]
temp_fin = temp_fin[temp_fin['Последняя температура'] > 1450]
print('Первый замер', temp_fin['Первая температура'].min())
print('Ппоследний замер', temp_fin['Последняя температура'].min())


# In[73]:


temp_fin.isnull().sum()


# In[74]:


temp_fin.info()


# Создал маблицу по первым и финальным температурам каждоко ковша, удалил аномально низкие значения температуры.

# **Таблица wire**

# In[75]:


df_wire = df_wire.fillna(0)
df_wire.head(10)


# In[76]:


df_wire['Sum Wire'] = df_wire.sum(axis=1)
#df_wire_new = df_wire.drop(['Wire 1', 'Wire 2', 'Wire 3', 'Wire 4', 'Wire 5', 'Wire 6', 'Wire 7', 'Wire 8', 'Wire 9'], axis=1)
df_wire.head(10)


# Так же как и с сыпучими материалами вывел в новую таблицу партии и суммарный обьем материалов

# **Обьеденим таблицы и подготовим финальную таблицу**

# In[77]:


df_final = (temp_fin.merge(df_arc_new, how='inner', on='key')
            .merge(df_bulk, how='inner', on='key')
            .merge(df_gas, how='inner', on='key').
            merge(df_wire, how='inner', on='key'))

df_final.head(10)


# In[78]:


df_final.isnull().sum()


# In[79]:


df_final.info()


# In[80]:


df_final = df_final.drop(['key'], axis=1)


# Обьеденил таблицы в одну таблицу

# In[81]:


df_final.describe()


# In[82]:


for i in df_final.columns:
    sns.boxplot(data=df_final[i])
    plt.title(i)
    plt.show()


# In[83]:


for i in df_final.columns:
    probplot(x=df_final[i],dist='norm',plot=plt)
    plt.title(i)
    plt.show()


# In[84]:


df_final.hist(figsize=(10,10))
plt.show()


# **Разделяю данные на выборки и стандартизирую их**

# In[85]:


features = df_final.drop(['Последняя температура'], axis=1)
target = df_final['Последняя температура']
features_train, features_test ,target_train, target_test = train_test_split(
    features, target, test_size=0.25, random_state=80523
)


# In[87]:


scaler = StandardScaler().fit(features_train)
x_train = scaler.transform(features_train)
x_test = scaler.transform(features_test)


# In[88]:


features_train = pd.DataFrame(data=x_train, columns=features.columns)
features_test= pd.DataFrame(data=x_test, columns=features.columns)


# In[89]:


features_train


# In[90]:


print(features_train.corr())


# In[106]:


sns.set(rc = {'figure.figsize':(25,16)})
features_plot = sns.heatmap(features_train.corr(), cmap="YlGnBu", annot=True)
plt.show()


# Наблюдаем мультиколлинеарность на тепловой какрте, пока трогать не буду посмотрю график важности факторов.

# **Строим модели**

# In[92]:


adam = keras.optimizers.Adam(lr=0.001)
steps_per_epoch = len(features_train)
validation_steps = len(features_test)


# In[93]:


model_keras = Sequential()
model_keras.add(keras.layers.Dense(units=1, input_dim=features.shape[1]))
model_keras.add(Dense(units=12, activation='relu'))
model_keras.add(Dense(units=24, activation='relu'))
model_keras.add(Dense(units=32, activation='relu'))
model_keras.add(Dense(units=64, activation='relu'))
model_keras.add(Dense(units=1, activation='relu'))
model_keras.compile(optimizer='sgd', loss=MeanSquaredError(), metrics=[MeanAbsoluteError()])
model_keras.fit(features_train, target_train, epochs=10, validation_split=0.1)


# In[94]:


predicted = model_keras.predict(features_train)


# In[95]:


print(mean_absolute_error(target_train, predicted))


# In[ ]:


get_ipython().run_cell_magic('time', '', "rfr = RandomForestRegressor(random_state=80523, n_jobs = -1)\nparam = {'max_depth':[15, 30, 40, None], \n         'n_estimators':[500, 600, 700], \n         'max_features':[13, 15, 17, 19, 21]\n        }\n\nparam_rfr = GridSearchCV(rfr, param_grid=param, cv=10, scoring='neg_mean_absolute_error')\nmodel_rfr = param_rfr.fit(features_train, target_train)\nprint(model_rfr.best_params_)\nprint(model_rfr.best_score_)")


# In[ ]:


predicted_rfr = model_rfr.predict(features_train)
print(mean_absolute_error(target_train, predicted_rfr))


# In[97]:


get_ipython().run_cell_magic('time', '', "model_cat = CatBoostRegressor(iterations=100, depth=16, loss_function='RMSE',random_state = 80523)\nmodel_cat.fit(features_train, target_train, verbose=10)")


# In[98]:


predicted_cat = model_cat.predict(features_train)
print(mean_absolute_error(target_train, predicted_cat))


# In[99]:


model_lgbm = LGBMRegressor(metric='mae',random_state = 80523,n_jobs = -1,learning_rate = 0.05,
                           silent = True, max_depth=30, num_leaves=20
                          )

model_lgbm.fit(features_train,target_train)
predict = model_lgbm.predict(features_train)
print('MAE:',mean_absolute_error(target_train, predict))


# In[ ]:


param = {'max_depth':[15, 30, 40, None], 
         'num_leaves':[10, 20, 30, 40,50], 
         'learning_rate':[0.1, 0.05, 0.4, 0.4, 0.7, 0.3]
        }
model_lgbm = LGBMRegressor(metric='mae', random_state = 80523, n_jobs = -1, silent = True)

param_lgbm  = GridSearchCV(model_lgbm, param_grid=param, cv=10, scoring='neg_mean_absolute_error')
model_lgbm  = param_lgbm.fit(features_train, target_train)
print(model_lgbm.best_params_)
print(model_lgbm.best_score_)

predicted_lgbm = model_lgbm.predict(features_train)
print(mean_absolute_error(target_train, predicted_lgbm))


# {'learning_rate': 0.05, 'max_depth': 30, 'num_leaves': 20}
# -6.092620909825144
# 4.232383702705017

# **Лучшая модель**

# Лучшей моделью показала себя model_lgbm её и будем использовать на тестовой выборке

# In[107]:


predicted_lgbm_test = model_lgbm.predict(features_test)
print(mean_absolute_error(target_test, predicted_lgbm_test))


# **Проверка на адекватность модели**

# In[101]:


dummy_regr = DummyRegressor(strategy="mean")
dummy_regr.fit(features_train, target_train)
predict_dummy = dummy_regr.predict(features_train)
print(mean_absolute_error(target_train, predict_dummy))


# **Важность признаков**

# In[102]:


importances = model_cat.feature_importances_
indices = np.argsort(importances)

fig, ax = plt.subplots(figsize=(10,20))
ax.barh(range(len(importances)), importances[indices])
ax.set_yticks(range(len(importances)))
_ = ax.set_yticklabels(np.array(features_train.columns)[indices])
ax.set_title("Важность признаков")
ax.set_xlabel('Признаки')
ax.set_xlabel('Важность')
plt.show()


# ![image.png](attachment:image.png)

# # Отчёт

# ## **План работы**
# - произвести обработку данных(пункт плана был выполнен)
# - соеденить таблицы в одну методом merge по столбцу 'key'(пункт плана был выполнен)
# - разделить данные на обучающую валидационную и тестовую выборки(пункт плана был выполнен)
# - произвести разделение на признаки и целевой признак(пункт плана был выполнен)
# - произвети обучение нескольких моделей регрессии(пункт плана был выполнен)
# - выбрать модель с лучшими показателями(пункт плана был выполнен)
# - проверить её на тестовой выборке и на адекватность(пункт плана был выполнен)

# ## 
# Трудности возникли с обучением модели т.к. неправельно был выведен целевой признак, после коментария тимлида обратил на это внимание исправил и всё заработало.

# ## 
# Ключевые шаги выделяю такие как
# - изучение данных
# - уточнение задачи
# - обработка данных и добавение новых признаков
# - создание модели 
# - провека модели

# ## 
# Лучшей моделью у меня является модель градиентного бустинга Light Gradient Boosted Machine (LightGBM) Regressor, 
# с качеством на тестовой выборке МАЕ = 6.25

# # Признаки

# In[100]:


for row in features_train.columns:
    display(row)


# Обработал признаки на пропуски удалив ковши где пропущены данные последней температуры, у остальных признаков заменил пропуски на 0.
# Удалил вбросы по температуре убрав ковши у которых температура ниже 1450 градусов верхние вбросы не трогал.
# Из 'Активная мощность' и 'Реактивная мощность' мощности создал признак 'Полная мощность' и 'Потребляемая мощьность'.
# Так же создал признаки 'Кол-во запусков', 'Время нагрева дугой', 'Sum Wire', 'Sum Bulk' и 'Первая температура'.
# Создал целевой признак.

# # Гиперпараметры

# metric = 'mae', random_state = 80523, n_jobs = -1, learning_rate = 0.05, max_depth=30, num_leaves=20

# # Рекомендации

# Можно попробовать убрать проблему мультиколлинеарности, добавить новые фичи и убрать признаки которые неимеют важности.

# Поместим очищенный датасет в переменную `cleaned_dataset` и сделаем группировку.

# Можно попробовать убрать проблему `мультиколлинеарности` 
