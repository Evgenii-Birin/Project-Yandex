#!/usr/bin/env python
# coding: utf-8

# # Сборный проэкт.

# Вы работаете в интернет-магазине «Стримчик», который продаёт по всему миру компьютерные игры. Из открытых источников доступны исторические данные о продажах игр, оценки пользователей и экспертов, жанры и платформы (например, Xbox или PlayStation). Вам нужно выявить определяющие успешность игры закономерности. Это позволит сделать ставку на потенциально популярный продукт и спланировать рекламные кампании.
# Нужно выявить определяющие успешность игры закономерности. Это позволит сделать ставку на потенциально популярный продукт и спланировать рекламные кампании.

# ### Описание данных

# Name — название игры
# Platform — платформа
# Year_of_Release — год выпуска
# Genre — жанр игры
# NA_sales — продажи в Северной Америке (миллионы проданных копий)
# EU_sales — продажи в Европе (миллионы проданных копий)
# JP_sales — продажи в Японии (миллионы проданных копий)
# Other_sales — продажи в других странах (миллионы проданных копий)
# Critic_Score — оценка критиков (максимум 100)
# User_Score — оценка пользователей (максимум 10)
# Rating — рейтинг от организации ESRB (англ. Entertainment Software Rating Board). Эта ассоциация определяет рейтинг компьютерных игр и присваивает им подходящую возрастную категорию.

# ##  Откроем файл с данными и изучим общую информацию.

# In[304]:


import pandas as pd 
df = pd.read_csv('/datasets/games.csv')
df.info()


# Нужно заменить буквы верхнего регистра в названиях столбцов на нижний регистр

# In[305]:


df.head(10)


# Есть пропущеные значения. 
# Нужно преобразовать столбец 'year_of_release' в тип Float

# ## Подготовим данные

# In[306]:


df.columns = df.columns.str.lower()
df.columns


# заменил названия сторбцов на коректные в нижнем регистре

# In[307]:


df['user_score'] = pd.to_numeric(df['user_score'], errors='coerce')


# перевожу в числовой тип данных, т.к. оценки пользователей это цифровой формат

# In[308]:


df.isna().sum()


# пропуски в столбцах: name, year_of_release, critic_core, user_score, rating

# In[309]:


df['name'] = df['name'].fillna(0)
df = df.loc[df['name'] !=0]
df.isna().sum()


# удалил строки с пропущеными значениями названия игр

# In[310]:


df['year_of_release'] = df['year_of_release'].fillna(0)
df = df.loc[df['year_of_release'] !=0]
df['year_of_release'] = df['year_of_release'].astype(int)
df.info()


# удалил пропущеные значения т.к. в последствии они могут помешать анализу и перевёл в тип int

# In[311]:


df.isna().sum()


# значения столбцов critic_core и user_score оставил без изменения, так как такое большое количество пропусков призамене могут исказить данные при анализе.
# значения tbd (TBD - аббревиатура от английского To Be Determined (будет определено) или To Be Decided (будет решено)) тоже оставил без изменения так как оценка похоже на тот момент была неопределена

# In[312]:


df['rating'].unique()


# In[313]:


import numpy as np
def replace_rating(rating_ka, rating_e):
    df['rating'] = df['rating'].replace(rating_ka, rating_e)
replace_rating(np.NaN, 'неизвестен')
replace_rating('K-A', 'E')

df['rating'].unique()


# заменил устаревшее обозначение рейтинга 'К-А' на современное 'Е'(KA для детей и взрослых: Игры, содержащие контент, подходящий для широкой аудитории. Этот рейтинг использовался до 1998 года, когда он был переименован в E), и значение NaN на 'неизвестен'

# In[314]:


df.isna().sum()


# проверил пропусков в данных нет

# In[315]:


df['sum_sales'] =  df[['na_sales','eu_sales','jp_sales', 'other_sales']].sum(axis = 1)
df.head()


# добавил столбец с сумарными продажами во всех регионах

# ## Проведём исследовательский анализ данных

# In[316]:


count_games = df.groupby('year_of_release')['name'].count()
display(count_games)


# до 94 года выпускалось не очень большое количество игр, с 94 выпуск игр возрастал больше всего игр вышло с 2005 до 2011 год, скорее всего по причине развития индустрии игр и их сравнительно небольшого размера. С 2012 производство игр упало до 500-600 игр в год скорее всего это связано с тем что игры стали технически сложнее улучшилась графика вследствии чего увеличилось время разработки новых игр.
# Для дальнейшего расмотрения я бы взял период с 2013 по 2016 год включительно

# In[317]:


import matplotlib.pyplot as plt
df.pivot_table(index='year_of_release', columns = 'platform', values='sum_sales', aggfunc='sum').plot(grid=True, figsize=(15, 10))
plt.title('График изменения продаж игр по платформам')
plt.xlabel('Год выпуска')
plt.ylabel('Количество продаж')
plt.show()


# по графику видно что платформы живут от 5 до 15 лет

# In[318]:


data = df.query("2013 <= year_of_release <= 2016")


# In[319]:


data.pivot_table(index='year_of_release', columns = 'platform', values='sum_sales', aggfunc='sum').plot(grid=True, figsize=(15, 10))
plt.title('График изменения продаж игр по платформам за 2013-2016г.г. ')
plt.xlabel('Год выпуска')
plt.ylabel('Количество продаж')
plt.show()


# по графику видно что продажи на всех платформах просели. продажи на РСР с 2015 года вообще близки к 0. 
# стабильно держится РС хотя продажи игр там сравнительно не велики.
# больше всего продаж на PS4, XBOX ONE и Nindendo 3DS
# 

# In[320]:


display(data.groupby(['platform'])['na_sales', 'eu_sales', 'jp_sales', 'other_sales', 'sum_sales'].sum())


# таблитца потверждает выводы по графику, но показывает большие продажи на РС3 но как видно из графика большие зачения продаж эта платформа имела в 2013 году и к 2015 году продажи упали

# In[321]:


data_2 = data.query("platform == ['XOne', 'PS4']")
display(data_2)


# Выбрал 2  потенциально прибыльных платформы опираясь на данные графика и таблицы

# In[322]:


import seaborn as sns
data_2.groupby('platform')['sum_sales'].describe()


# In[323]:


data_2.boxplot(column='sum_sales', by='platform', figsize=(15,10), whis=15)
plt.title('График «ящик с усами» по глобальным продажам каждой игры и разбивкой по платформам за период 2013 - 2016гг.')
plt.xlabel('Наименование игровой платформы')
plt.ylabel('Продажи игр')
plt.show()

fig, boxplot = plt.subplots(figsize = (15,10))
boxplot = sns.boxplot(x='sum_sales', y='platform', data=data_2, width=0.9, whis=15)
boxplot.axes.set_title('График «ящик с усами» по глобальным продажам каждой игры и разбивкой по платформам за период 2013 - 2016гг.', fontsize=15)
boxplot.set_ylabel('Наименование игровой платформы', fontsize=15)
boxplot.set_xlabel('Продажи игр', fontsize=15)


# есть много выбросов особенно большой разброс у платформы РС4, это может быть связано с играми которые получили очень большую популярность, больше всего вбросов у платформы РС4 так что для дальнейшего анализа возьму её

# In[324]:


data_2[data_2['platform']=='PS4'].plot(x='user_score', y='sum_sales', kind='scatter', alpha=0.9, figsize=(15,7), grid=True)
plt.title('Диаграмма рассеивания')
plt.xlabel('Отзывы пользователей')
plt.ylabel('Сумма продаж')
plt.show()


# вывел таблицу рассеивания по платформе PS4, отобразив отзывы пользователей

# In[325]:


data_2[data_2['platform']=='PS4'].plot(x='critic_score', y='sum_sales', kind='scatter', alpha=0.9, figsize=(15,7), grid=True)
plt.title('Диаграмма рассеивания')
plt.xlabel('Отзывы критиков')
plt.ylabel('Сумма продаж')
plt.show()


# вывел таблицу рассеивания по платформе PS4, отобразив отзывы критиков

# In[326]:


data_pc4 = data_2[data_2['platform']=='PS4']
print(data_pc4['user_score'].corr(data['sum_sales']))
print(data_pc4['critic_score'].corr(data['sum_sales']))


# кореляция между отзывами пользователей и продажами очень лабая и скорее всего эти отзывы не влияют на продажи
# кореляция между оценками критиков и продажами сильнее че у пользователей но тоже слабая

# In[327]:


data.groupby('platform')['sum_sales'].describe()


# In[328]:


data.boxplot(column='sum_sales', by='platform', figsize=(15,7))
plt.ylim(0, 4)
plt.title('Диаграмма размаха')
plt.xlabel('Отзывы критиков')
plt.ylabel('Сумма продаж')
plt.show()


# из графика видно что больше всего продаж в данный период времени было у 4 платформ это РС4, Nintendo Wii, Xbox 360 и XboxOne

# In[329]:


data.pivot_table(index='genre', values='sum_sales', aggfunc='median').boxplot(column='sum_sales', by='genre', figsize=(15,7))
plt.title('Общее распределение игр по жанрам')
plt.xlabel('Жанр')
plt.ylabel('Количество продаж')
plt.show()


# самые популярные жанры в нашей выборке Platform и Sports и Shooter
# самые низкие продажи у Adventure и Puzzie

# ## Составим портрет пользователя каждого региона

# In[330]:


platform_na = data.groupby('platform')['na_sales'].sum().sort_values(ascending=False).head(5)
platform_na.plot(x = 'platform', y = 'na_sales', kind = 'bar', figsize=(15,7), grid=True)
plt.title('5 самых популярных платформ в Северной Америке')
plt.xlabel('Платформа')
plt.ylabel('Продажи игр')
plt.show()


# In[331]:


platform_full = data.groupby('platform')['na_sales'].sum().sort_values(ascending=False).head(5)
print(
    'Доля продаж самой популярной платформы РС4 в Северной Америке по отношению к другим платформам', 
    platform_full['PS4'] / platform_full.sum() *100, '%'
)
print(platform_full)


# In[332]:


platform_eu = data.groupby('platform')['eu_sales'].sum().sort_values(ascending=False).head(5)
platform_eu.plot(x = 'platform', y = 'eu_sales', kind = 'bar', figsize=(15,7), grid=True)
plt.title('5 самых популярных платформ в Европе')
plt.xlabel('Платформа')
plt.ylabel('Продажи игр')
plt.show()


# In[333]:


platform_full = data.groupby('platform')['eu_sales'].sum().sort_values(ascending=False).head(5)
print(
    'Доля продаж самой популярной платформы РС4 в Европе по отношению к другим платформам', 
    platform_full['PS4'] / platform_full.sum() *100, '%'
)
print(platform_full)


# In[334]:


platform_jp = data.groupby('platform')['jp_sales'].sum().sort_values(ascending=False).head(5)
platform_jp.plot(x = 'platform', y = 'jp_sales', kind = 'bar', figsize=(15,7), grid=True)
plt.title('5 самых популярных платформ в Японии')
plt.xlabel('Платформа')
plt.ylabel('Продажи игр')
plt.show()


# In[335]:


platform_full = data.groupby('platform')['jp_sales'].sum().sort_values(ascending=False).head(5)
print(
    'Доля продаж самой популярной платформы 3DS в Европе по отношению к другим платформам', 
    platform_full['3DS'] / platform_full.sum() *100, '%'
)
print(platform_full)


# в европе и северной америке самой популярной платформой является РС4 а в японии nindendo DS

# In[336]:


genre_na = data.groupby('genre')['na_sales'].sum().sort_values(ascending=False).head(5)
genre_na.plot(x = 'genre', y = 'na_sales', kind = 'bar', figsize=(15,7), grid=True)
plt.title('5 самых популярных жанров в Северной Америке')
plt.xlabel('Жанр')
plt.ylabel('Продажи игр')
plt.show()


# самыми популярными жанрами в северной америке являются экшин и шутер жанры где много движения и сражений

# In[337]:


genre_eu = data.groupby('genre')['eu_sales'].sum().sort_values(ascending=False).head(5)
genre_eu.plot(x = 'genre', y = 'eu_sales', kind = 'bar', figsize=(15,7), grid=True)
plt.title('5 самых популярных жанров в Европе')
plt.xlabel('Жанр')
plt.ylabel('Продажи игр')
plt.show()


# в европе такая же картина как и в северной америке

# In[338]:


genre_jp = data.groupby('genre')['jp_sales'].sum().sort_values(ascending=False).head(5)
genre_jp.plot(x = 'genre', y = 'jp_sales', kind = 'bar', figsize=(15,7), grid=True)
plt.title('5 самых популярных жанров в Японии')
plt.xlabel('Жанр')
plt.ylabel('Продажи игр')
plt.show()


# а в японии на первом месте стот ролевыи игры а на втором экшен

# In[339]:


rating_na = data.groupby('rating')['na_sales'].sum().sort_values(ascending=False).head(5)
rating_na.plot(x = 'rating', y = 'na_sales', kind = 'bar', figsize=(15,7), grid=True)
plt.title('Рейтинг от организации ESRB в Северной Америке')
plt.xlabel('Рейтинг')
plt.ylabel('Продажи игр')
plt.show()


# в северной америке больше всего предпочитают игры для взрослых с рейтингом 17+

# In[340]:


rating_eu = data.groupby('rating')['eu_sales'].sum().sort_values(ascending=False).head(5)
rating_eu.plot(x = 'rating', y = 'eu_sales', kind = 'bar', figsize=(15,7), grid=True)
plt.title('Рейтинг от организации ESRB в Европе')
plt.xlabel('Рейтинг')
plt.ylabel('Продажи игр')
plt.show()


# в европе также предпочитают игра с рейтингом 17+

# In[341]:


rating_jp = data.groupby('rating')['jp_sales'].sum().sort_values(ascending=False).head(5)
rating_jp.plot(x = 'rating', y = 'jp_sales', kind = 'bar', figsize=(15,7), grid=True)
plt.title('Рейтинг от организации ESRB в Японии')
plt.xlabel('Рейтинг')
plt.ylabel('Продажи игр')
plt.show()


# а в японии болшинство предпочитают с неопределенным рейтингом

# ## Проверка гипотез

# Для проверки гипотезы я использую st.ttest_ind так как нужно проверить гипотизу о равенстве среднего двух генеральных совокупностей по взятым из них выборкам. Выборки не зависят друг от друга (не являются парными) так как значения из оной выборки не как не зависят от значений другой выборки.  

# Для проверки гипотезы "средние пользовательские рейтинги платформ Xbox One и PC одинаковые" в качестве нулевой и альтернативной гипотезы мы взяли следующее:
# H0: средние рейтинги по платформам одинаковые
# H1: средние рейтинги по платформам разные
# Для проверки гипотезы я использую st.ttest_ind так как нужно проверить гипотизу о равенстве среднего двух генеральных совокупностей по взятым из них выборкам. Выборки не зависят друг от друга (не являются парными) так как значения из оной выборки не как не зависят от значений другой выборки.

# In[342]:


from scipy import stats as st
data['user_score'] = data['user_score'].fillna(-1)
data = data[data['user_score'] != -1]

data_xbox = data[data['platform'] == 'XOne']
data_pc = data[data['platform'] == 'PC']

xbox_1 = data_xbox['user_score']
pc_1 = data_pc['user_score']
alpha = .05

results = st.ttest_ind(xbox_1, pc_1)

print('p-значение:', results.pvalue)

if (results.pvalue < alpha):
    print("Отвергаем нулевую гипотезу")
else:
    print("Не получилось отвергнуть нулевую гипотезу")


# Для проверки гипотезы "средние пользовательские рейтинги жанров Action и Sports разные" в качестве нулевой и альтернативной гипотезы мы взяли следующее:
# H0: средние рейтинги по жанрам одинаковые
# H1: средние рейтинги по жанрам разные
# Для проверки гипотезы я использую st.ttest_ind так как нужно проверить гипотизу о равенстве среднего двух генеральных совокупностей по взятым из них выборкам. Выборки не зависят друг от друга (не являются парными) так как значения из оной выборки не как не зависят от значений другой выборки.

# In[343]:


data = data[data['user_score'] != -1]
data_action = data[data['genre'] == 'Action']
data_sports = data[data['genre'] == 'Sports']
action_1 = data_action['user_score']
sports_1 = data_sports['user_score']
alpha = .05

results = st.ttest_ind(action_1,sports_1)

print('p-значение: ', results.pvalue)
if results.pvalue < alpha:
    print("Отвергаем нулевую гипотезу")
else:
    print("Не получилось отвергнуть нулевую гипотезу")


# ## Вывод

# По итогам можно сделать прогноз на 2017 год в игровой индустрии.
# 
# Был анализирован период с 2013-2016 года, и в этот период для диагностики у нас были выбраны 2 платформы: PS4 и Xbox One, которые на этот момент обладали самыми большими продажами и были в тренде.
# 
# Самыми популярными платформами на рынке в мире являются - PS4, т.к. из из всех стран, только Япония склоняется к Nintendo 3DS.
# 
# Жанры игр страны предпочитают разные, но большинство склоняется к жанру Action. Также в ТОП жанров попали такие как: Shooter (в Северной Америке и Европе) и Role-Playing (в Японии).
# 
# А вот ТОП рейтинг от организации ESRB можно твердо считать категорию M («Для взрослых»: Материалы игры не подходят для подростков младше 17 лет).
# 
# На следующий год ставку можно делать на платформу PS4, т.к. она популярна в большинстве стран мира. В жанрах предложил бы сделать упор на Action, но не забывать про остальные два жанра, т.к. они тоже имеют популярность. В возрастной группе предложил бы сделать ставку на категорию 17+.
# 
# 
# 
