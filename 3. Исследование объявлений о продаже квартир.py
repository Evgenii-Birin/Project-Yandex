#!/usr/bin/env python
# coding: utf-8

# # Исследование объявлений о продаже квартир
# 
# В вашем распоряжении данные сервиса Яндекс.Недвижимость — архив объявлений о продаже квартир в Санкт-Петербурге и соседних населённых пунктов за несколько лет. Нужно научиться определять рыночную стоимость объектов недвижимости. Ваша задача — установить параметры. Это позволит построить автоматизированную систему: она отследит аномалии и мошенническую деятельность. 
# 
# По каждой квартире на продажу доступны два вида данных. Первые вписаны пользователем, вторые — получены автоматически на основе картографических данных. Например, расстояние до центра, аэропорта, ближайшего парка и водоёма. 

# ### Откройте файл с данными и изучите общую информацию. 

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv('/datasets/real_estate_data.csv', sep='\t')
data.info()
display(data)
data.hist(figsize=(15, 20))


# 

# фрейм с данными состоит из 23699 строк и из 22 столбцов
# после открытия фрейма данных заметил следующие проблемы: нужно поменять тип данных в некоторых столбцах, есть попущеные значения типа NaN, нужно изменить некоректное название столбца 'cityCenters_nearest' поменять на 'city_centers_nearest'.

# ### Предобработка данных

# In[2]:


data.rename(columns={'cityCenters_nearest': 'city_centers_nearest'}, inplace=True) 
data.isna().sum()
data['first_day_exposition'] = pd.to_datetime(data['first_day_exposition'], format = '%Y-%m-%d')
data[data['floors_total'].isna()]
data['is_apartment'] = data['is_apartment'].fillna(False).astype('bool')


# приведем значение столбца к коректному виду
# определим в каких столбцах пропущенные значения
# перевел столбец в нужный формат, время указывать не стал т.к. оно везде 0
# закономерности в пропуске данных не вижу т.к. помещения находятся на разных этажах и с разной площадью, поэтому пропуски скорее всего просто не заполнены оставляю столбец в таком виде
# заменил все пропущеные значения на False и сделал столбец типа, bool

# In[3]:


data['parks_around3000'].isna().sum()
data['parks_around3000'].value_counts()
data['parks_around3000'] = data['parks_around3000'].fillna(0).astype('int')


# заменил пропущеные значения на ноль, т.к. данные были не заполнены по причине отсутствия парков в радиусе 3 км, и перевел в Int

# In[4]:


data['ponds_around3000'].isna().sum()
data['ponds_around3000'].value_counts()
data['ponds_around3000'] = data['ponds_around3000'].fillna(0).astype('int')


# сделал тоже что и с парками обоснавание тоже

# In[5]:


data['days_exposition'].isna().sum()
data['days_exposition'].value_counts()
data['days_exposition'] = data['days_exposition'].astype('int', errors='ignore')


# перевёл в тип int

# In[6]:


data['ceiling_height'].value_counts()
data['ceiling_height'] = data['ceiling_height'].fillna(data['ceiling_height'].median())


# заменил значения на медианные

# In[7]:


data['balcony'].value_counts()
data['balcony'] = data['balcony'].fillna(0)


# можно предположить что в пропущеных значениях отсуиствуют балконыб заменил отсутствуещие значения на 0

# In[8]:


data['living_area'].value_counts()


# можно попробовать высчитать значение из разности общей площади и площади кухни но не везде есть данные трогать не буду

# In[9]:


data['kitchen_area'].value_counts()


# причина пропусков не ясна скорее всего просто не указаны данные

# In[10]:


data['airports_nearest'].value_counts()


# пропущеные значения трогать не буду

# In[11]:


data['city_centers_nearest'].value_counts()

data['parks_nearest'].value_counts() 

data['ponds_nearest'].value_counts()


# не трогаю

# In[12]:


data['locality_name'] = data['locality_name'].str.lower()
data['locality_name'] = data['locality_name'].str.replace('ё', 'е', regex=True)
locality_name_unique = data['locality_name'].unique()


# перевёл названия в нижний регистр и заменил все буквы ё на е

# In[13]:


display(locality_name_unique)
print(len(locality_name_unique))


# количество уникальных значений уменьшилось

# In[14]:


data = data.dropna(subset=['locality_name'])
display(locality_name_unique)


#  удалил строки с пропущеными значениями городов

# In[15]:


sorted_ceiling_height = data['ceiling_height'].sort_values()
data = data[(data['ceiling_height'] > 2) & (data['ceiling_height'] < 100)]
data.loc[(data['ceiling_height'] > 20) & (data['ceiling_height'] <= 32) , 'ceiling_height'] /= 10
data = data[(data['ceiling_height'] > 2) & (data['ceiling_height'] < 5)]
sorted_ceiling_height = data['ceiling_height'].sort_values()
display(sorted_ceiling_height)


# привел значения высоты потолков к коректному видуб удалил анамальные значения

# In[16]:


data.describe()


# поменял тип некоторых столбцов удалил некоторые строки с отсутствующими и некоректными значениями

# first_day_exposition - формат столбца типа object, хотя должен быть data.time
# floors_total - формат столбца должен быть int, так как количество этажей к доме является целым числом.
# is_apartment - должно быть типа bool
# parks_around3000 и ponds_around3000 должно быть тип int, т.к. количество парков и водоемов целое число
# days_exposition - количество дней должно быть тип int
# Есть пропущеные значения
# Есть аномалии

# ### Посчитайте и добавьте в таблицу новые столбцы

# посчитаем цену квадратного метра и добавим столбец в таблицу

# In[17]:


data['price_meter'] = data['last_price'] / data['total_area']


# добовляю день недели
# месяц
# год

# In[18]:


data['weekday'] = data['first_day_exposition'].dt.weekday

data['month'] = data['first_day_exposition'].dt.month

data['year'] = data['first_day_exposition'].dt.year 


# функция категоризации этажей продаваемых квартир

# In[19]:


def floor(row):
    floors_total = row['floors_total']
    floor = row['floor']
    if floor == 1:
        return 'первый'
    elif floor == floors_total:
        return 'последний'
    elif 1 < floor < floors_total:
        return 'другой'


# применяю функцию

# In[20]:


data['floor_category'] = data.apply(floor, axis = 1)


# перевожу растояние от центра города в км

# In[21]:


data['city_centers_nearest'].dtypes # 
data['city_centers_nearest'] = data['city_centers_nearest'].astype('float64', errors='ignore')
data.loc[(data['city_centers_nearest'] != 'NaN'), 'city_centers_nearest']/1000
data['city_centers_nearest'] = data['city_centers_nearest'].round()


#display(data['city_centers_nearest'].head(50))
#data.info()


# проверяю тип столбца
# перевожу тип столбца в float
# перевожу значения в км
# привожу к целому

# ### Проведите исследовательский анализ данных

# In[22]:


# общая площадь
data.plot(y = 'total_area', kind = 'hist', bins = 600, grid=True, figsize = (15,7), range = (0,900)) # есть небольшие аномальные значения
plt.xlim(10, 100)
data['total_area'].describe()


# больше всего продают квартиры от 30 до 70 кв.м

# In[23]:


# код ревьюера

data['total_area'].hist(bins = 600, figsize = (15, 7), color='#A7D2CB')
plt.xlim(10, 100)

plt.title('Распределение значений общей площади')
plt.xlabel('Площадь') 
plt.ylabel('Количество объявлений')
plt.show()


# In[24]:


# жилая площадь
data.plot(y = 'living_area', kind = 'hist', bins = 50, grid=True, figsize = (15,7), range = (0,200)) # есть выбевающиеся значения
#plt.xlim(10, 100)
data['living_area'].describe()


# больше всего квартир с жилой площадью до 30 кв.м

# In[25]:


# площади кухни
data.plot(y = 'kitchen_area', kind = 'hist', bins = 50, grid=True, figsize = (15,7), range = (0,60)) # есть выбевающееся значения
data['kitchen_area'].describe()


# большенство продоваемых квартир с площадью кухни до 10 кв.м

# In[26]:


# количества комнат

data['rooms'] = data['rooms'].fillna(0) # заменил отсутствующие значения на 0 зачем то
data['rooms'] = data['rooms'].round() # привел к цетому т.к. комнаты 
data = data[data['rooms'] > 0] # убрал нулевые значения
data['rooms'] = data['rooms'].astype('int')
data.plot(y = 'rooms', kind = 'hist', bins =15, grid=True, figsize = (15,7))
plt.title('Распределение значений от количества комнат')
plt.xlabel('Количество комнат') 
plt.ylabel('Количество объявлений')
plt.show()
data['rooms'].describe()


# больше всего обьявлений о продаже 1 и 2-х комнатных квартир

# In[27]:


# этаж
category_floor = data.pivot_table(index = 'floor_category', values = 'floor', aggfunc = 'count')
data.plot(y='floor', kind='hist')
category_floor.plot.bar(use_index=True) # 
data['floor'].describe()
#display(category_floor)


# первый и последний этаж продаютс меньше чем остальные, больше всего обьявлений до 5 этажа

# In[28]:


# общее количество этажей в доме
data.plot(y = 'floors_total', kind = 'hist', bins = 30, range = (2,5), grid=True, figsize = (5,3))
data['floors_total'].describe()


# чаще всего квартиры продают в пятиэтажках 

# In[29]:


# расстояние до центра города
data = data[data['city_centers_nearest'] > 0]
data.plot(y = 'city_centers_nearest', kind = 'hist', bins = 70)
data['city_centers_nearest'].describe() #есть выбивающиеся значения удалю все нулевые значения что бы не искажали данные


# больше всего обьявлений о продаже в радиусе от 12 до 16 км от центра 

# In[30]:


# расстояние до ближайшего аэропорта
data.plot(y = 'airports_nearest', kind = 'hist', bins = 70)
data['airports_nearest'].describe()


# больше всего обьявлений на растоянии от 10 до 30 км от аэропорта

# In[31]:


# расстояние до ближайшего парка
data.plot(y = 'parks_nearest', kind = 'hist', bins = 70)
data['parks_nearest'].describe()


# больше всего обьявлений с указаным растоянием до ближайшего парка находятся на растоянии от200 до 700 м до ближайшего парка

# In[32]:


data.plot(y = 'weekday', kind = 'hist')
data.plot(y = 'month', kind = 'hist')
#data['month'].hist()
#data['weekday'].describe()


# чаще всего обьявления подапались в январе и декабре в будние дни 

# In[33]:


# срок продажи недвижимости
data.plot(y = 'days_exposition', kind = 'hist', bins = 100, grid = True, range = (1,200))
data['days_exposition'].describe()


# большенство квартир продавались в течении 25 дней

# In[34]:



data.plot(x='total_area', y = 'last_price', style = 'o', figsize = (15,7))
plt.title('Распределение зависимости цены квартиры от общей площади', size=17)
plt.xlabel('Общая площадь квартиры', size=13)
plt.ylabel('Цена', size=13)

#pivot_table_total_area.sort_values('median', ascending = False)
data['total_area'].corr(data['last_price'])


# зависимость очень слабая

# In[35]:


# код ревьюера

import seaborn as sns

plt.figure(figsize=(15, 7))

sns.scatterplot(data=data,  x='total_area', y='last_price', color='#AC7088')

plt.title('Распределение зависимости цены квартиры от общей площади', size=17)
plt.xlabel('Общая площадь квартиры', size=13)
plt.ylabel('Цена', size=13)

plt.show()


# In[36]:


# Изучим зависимость цены от жилой площади
pivot_table_living_area = data.pivot_table(index = 'living_area', values = 'last_price', aggfunc = ['mean', 'count', 'median'])
pivot_table_living_area.columns = ['mean', 'count', 'median']
pivot_table_living_area.plot(y = 'median', style = 'o')

pivot_table_living_area.sort_values('median', ascending = False)
data['living_area'].corr(data['price_meter'])


# зависимость очень слабая

# In[37]:


# Изучим зависимость цены от площади кухни
pivot_table_kitchen_area = data.pivot_table(index = 'kitchen_area', values = 'last_price', aggfunc = ['mean', 'count', 'median'])
pivot_table_kitchen_area.columns = ['mean', 'count', 'median']
pivot_table_kitchen_area.plot(y = 'median', style = 'o', figsize = (15,7))

pivot_table_kitchen_area.sort_values('median', ascending = False)
data['living_area'].corr(data['price_meter'])


# зависимость очень слабая

# In[38]:


#Изучим зависимость цены квадратного метра от числа комнат
pivot_table_rooms = data.pivot_table(index = 'rooms', values = 'price_meter', aggfunc = ['mean', 'count', 'median'])
pivot_table_rooms.columns = ['mean', 'count', 'median']
pivot_table_rooms.query('count > 50').plot(y = 'median')

pivot_table_rooms.query('count > 50').sort_values('median', ascending = False)

data['rooms'].corr(data['price_meter'])


# больше всего цена за квадратный метр у однокомнатных квартир и квартир где 7  комнат, ниже всего цена за квадратный метр у 3-х комнатных квартир

# In[39]:


# Изучим зависимость цены квадратного метра от этажа
pivot_table_floor_category = data.pivot_table(index = 'floor_category', values = 'price_meter', aggfunc = ['mean', 'count', 'median'])
pivot_table_floor_category.columns = ['mean', 'count', 'median']
pivot_table_floor_category.plot(y = 'median')


# ниже всего цена у квартир расположеных на первом этаже

# In[40]:


# Изучим зависимость цены квадратного метра от даты размещения
pivot_table_exposition = data.pivot_table(index = 'weekday', values = 'price_meter', aggfunc = ['mean', 'count', 'median'])
pivot_table_exposition.columns = ['mean', 'count', 'median']
pivot_table_exposition.plot(y = 'median', figsize = (15,7))
pivot_table_exposition.sort_values('median', ascending = False)# самая большая цена за квадратный метр была 27 августа 2015 года

#data['weekday'].hist(y = 'price_meter', bins = 600, figsize = (15, 7), color='#A7D2CB')
#plt.xlim(10, 100)

plt.title('Расмотрим изменение цены квадратного метра от дня недели публикации')
plt.xlabel('день недели') 
plt.ylabel('Цена')
plt.show()


# ниже всего цена в пятницу

# In[41]:


pivot_table_exposition_mouth = data.pivot_table(index = 'month', values = 'price_meter', aggfunc = ['mean', 'count', 'median'])
pivot_table_exposition_mouth.columns = ['mean', 'count', 'median']
pivot_table_exposition_mouth.plot(y = 'median', figsize = (15,7))
pivot_table_exposition_mouth.sort_values('median', ascending = False)

plt.title('Расмотрим изменение цены квадратного метра от месяца публикации')
plt.xlabel('Месяц') 
plt.ylabel('Цена')
plt.show()


# самая высокая цена в апреле и декабре

# In[42]:


pivot_table_exposition_mouth = data.pivot_table(index = 'year', values = 'price_meter', aggfunc = ['mean', 'count', 'median'])
pivot_table_exposition_mouth.columns = ['mean', 'count', 'median']
pivot_table_exposition_mouth.plot(y = 'median', figsize = (15,7))
pivot_table_exposition_mouth.sort_values('median', ascending = False)

plt.title('Расмотрим изменение цены квадратного метра от года публикации')
plt.xlabel('Год') 
plt.ylabel('Цена')
plt.show()


# в 2014 году цены на квартиры упали, а с 2016 года неизменно росли

# In[43]:



data.plot(y = 'days_exposition', kind = 'hist', bins = 100, figsize = (15,7), grid = True, range = (1,200))

plt.title('Изучим, как быстро продавались квартиры', size=17)
plt.xlabel('Количество дней', size=13)
plt.ylabel('Количество продаж', size=13)
data['days_exposition'].describe()


# среднее время продажи 182 день,а медианное 96. Быстрыми продажами модно считать продажи до 46 дней, а долгими более 200 дней

# In[44]:


# выполнено задание: "Посчитайте среднюю цену одного квадратного метра в 10 населённых пунктах с наибольшим числом объявлений. Выделите населённые пункты с самой высокой и низкой стоимостью квадратного метра.
locality_pivot_table = data.pivot_table(index = 'locality_name', values = 'price_meter', aggfunc=['count', 'mean'])
locality_pivot_table.columns = ['count', 'mean']
locality_pivot_table = locality_pivot_table.sort_values('count', ascending = False).head(10)
locality_pivot_table
#самая высокая стоимость
locality_pivot_table[locality_pivot_table['mean']==locality_pivot_table['mean'].max()]
#самая низкая стоимость
locality_pivot_table[locality_pivot_table['mean']==locality_pivot_table['mean'].min()]

display(locality_pivot_table)
display(locality_pivot_table[locality_pivot_table['mean']==locality_pivot_table['mean'].max()])
display(locality_pivot_table[locality_pivot_table['mean']==locality_pivot_table['mean'].min()])


# самая высокая стоимость квадратного метра в питере
# самая низкая цена в красном селе

# In[45]:


#"Ранее вы посчитали расстояние до центра в километрах. Теперь выделите квартиры в Санкт-Петербурге с помощью столбца `locality_name` и вычислите среднюю цену каждого километра.
pivot_table_km = data.query('locality_name == "санкт-петербург"').pivot_table(index = 'city_centers_nearest', values = 'price_meter', aggfunc = 'mean')
pivot_table_km.plot()
pivot_table_km # 


# судя по графику центр города можно считать радиус 7 км по мере удаления от центра города цена на квадратный метр снижается

# ### Общий вывод

# при обработки фрейма данных произвел первичный осмотр информации о фрейме, привел названия столбцов к коректному виду, поработа с дубликатами названия населенных пунктов, заменил на ноль пропуски в строках с незаполнеными данными о количестве балконов,   удалил строки с пропущеными названиями населённых пунктов, построил графики на заданные значения, иследование показало что самая высокая стоимость у квартир имеющих 1 комнату или 7 комнат, на первом и последнем этаже цена за метр ниже, больше всего обьявлений в питере и самая большая цена за квадратный метр, самая низкая цена в населенном пункте красное село, цена в питере зависит от удаленности от центра города чем дальше от центра тем дешевле цена за квадратный метр что логично, обьявления висят в среднем 170 дней, быстрыми продажами можно считать проданую недвижимость в течении 46 дней, хотя есть и долгие обьявления больше 200 дней. цена квадратного метра очень слабо зависит от площади кухни, жилой площади и общей площади. 
# 
