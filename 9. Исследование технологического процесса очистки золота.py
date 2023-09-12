#!/usr/bin/env python
# coding: utf-8

# # Восстановление золота из руды

# Подготовьте прототип модели машинного обучения для «Цифры». Компания разрабатывает решения для эффективной работы промышленных предприятий.
# 
# Модель должна предсказать коэффициент восстановления золота из золотосодержащей руды. Используйте данные с параметрами добычи и очистки. 
# 
# Модель поможет оптимизировать производство, чтобы не запускать предприятие с убыточными характеристиками.
# 
# Вам нужно:
# 
# 1. Подготовить данные;
# 2. Провести исследовательский анализ данных;
# 3. Построить и обучить модель.
# 
# Чтобы выполнить проект, обращайтесь к библиотекам *pandas*, *matplotlib* и *sklearn.* Вам поможет их документация.

# <h1>Содержание<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Подготовка-данных" data-toc-modified-id="Подготовка-данных-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Подготовка данных</a></span><ul class="toc-item"><li><span><a href="#Откроем-файлы-и-изучим-их." data-toc-modified-id="Откроем-файлы-и-изучим-их.-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Откроем файлы и изучим их.</a></span></li><li><span><a href="#Проверим,-что-эффективность-обогащения-рассчитана-правильно." data-toc-modified-id="Проверим,-что-эффективность-обогащения-рассчитана-правильно.-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Проверим, что эффективность обогащения рассчитана правильно.</a></span></li><li><span><a href="#Проанализируем-признаки,-недоступные-в-тестовой-выборке." data-toc-modified-id="Проанализируем-признаки,-недоступные-в-тестовой-выборке.-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Проанализируем признаки, недоступные в тестовой выборке.</a></span></li><li><span><a href="#Проведите-предобработку-данных." data-toc-modified-id="Проведите-предобработку-данных.-1.4"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>Проведите предобработку данных.</a></span></li></ul></li><li><span><a href="#Анализ-данных" data-toc-modified-id="Анализ-данных-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Анализ данных</a></span><ul class="toc-item"><li><span><a href="#Посмотрим,-как-меняется-концентрация-металлов-(Au,-Ag,-Pb)-на-различных-этапах-очистки." data-toc-modified-id="Посмотрим,-как-меняется-концентрация-металлов-(Au,-Ag,-Pb)-на-различных-этапах-очистки.-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Посмотрим, как меняется концентрация металлов (Au, Ag, Pb) на различных этапах очистки.</a></span></li><li><span><a href="#Сравним-распределение-размеров-гранул-сырья-на-обучающей-и-тестовой-выборках." data-toc-modified-id="Сравним-распределение-размеров-гранул-сырья-на-обучающей-и-тестовой-выборках.-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Сравним распределение размеров гранул сырья на обучающей и тестовой выборках.</a></span></li><li><span><a href="#Исследуем-суммарную-концентрацию-всех-веществ-на-разных-стадиях." data-toc-modified-id="Исследуем-суммарную-концентрацию-всех-веществ-на-разных-стадиях.-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Исследуем суммарную концентрацию всех веществ на разных стадиях.</a></span></li></ul></li><li><span><a href="#Модель" data-toc-modified-id="Модель-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Модель</a></span><ul class="toc-item"><li><span><a href="#Напишим-функцию-для-вычисления-итоговой-sMAPE." data-toc-modified-id="Напишим-функцию-для-вычисления-итоговой-sMAPE.-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Напишим функцию для вычисления итоговой sMAPE.</a></span></li><li><span><a href="#Обучим-разные-модели-и-оценим-их-качество-кросс-валидацией.-Выберем-лучшую-модель-и-проверим-её-на-тестовой-выборке." data-toc-modified-id="Обучим-разные-модели-и-оценим-их-качество-кросс-валидацией.-Выберем-лучшую-модель-и-проверим-её-на-тестовой-выборке.-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Обучим разные модели и оценим их качество кросс-валидацией. Выберем лучшую модель и проверим её на тестовой выборке.</a></span></li></ul></li></ul></div>

# ## Подготовка данных

# ### Откроем файлы и изучим их.

# In[60]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from numpy.random import RandomState

from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegressionCV
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GridSearchCV


# In[61]:


data_train =  pd.read_csv('/datasets/gold_recovery_train_new.csv')
data_test = pd.read_csv('/datasets/gold_recovery_test_new.csv')
data_full = pd.read_csv('/datasets/gold_recovery_full_new.csv')


# Загрузил данные

# In[62]:


data_train.shape, data_test.shape, data_full.shape


# Вывел размеры фрейма данных

# In[63]:


data_train.info(), data_test.info(), data_full.info()


# Вывел информацию о фреймах данных

# In[64]:


display(data_train.head(), data_test.head(), data_full.head())


# Вывел первые пять строк фреймов данных

# In[65]:


print('Количество явных дубликатов в 1-ом датасете - ',data_train.duplicated().sum())
print('Количество явных дубликатов в 2-ом датасете - ',data_test.duplicated().sum())
print('Количество явных дубликатов в 3-ом датасете - ',data_full.duplicated().sum())


# In[66]:


data_train = data_train.ffill(axis=0)
data_test = data_test.ffill(axis=0)
data_full = data_full.ffill(axis=0)
print('Количество строк без пропусков в data_train - ', len(data_train.isnull().sum (axis= 1 )))
print('Количество строк без пропусков в data_test -  ', len(data_test.isnull().sum (axis= 1 )))
print('Количество строк без пропусков в data_full -  ', len(data_full.isnull().sum (axis= 1 )))


# С помощью функции ffill заполним отсутствующие значения в тренировочной и тестовой выборках. Функция заполняет пропуски предпоследним значением признака

# ### Проверим, что эффективность обогащения рассчитана правильно.

# In[67]:


c = data_train['rougher.output.concentrate_au']
f = data_train['rougher.input.feed_au']
t = data_train['rougher.output.tail_au']
recovery = ((c*(f-t))/(f*(c-t))) * 100

print('MAE -', mean_absolute_error(data_train['rougher.output.recovery'], recovery))


# Полученное значение мало, это значит что расчеты верны.

# ### Проанализируем признаки, недоступные в тестовой выборке.

# In[68]:


missing_columns = []
for name in data_train.columns:
    if name in data_test.columns:
        continue
    else:
        missing_columns.append(name)
print('Количество отсутствуещих признаков =', len(missing_columns))
print()
display(*missing_columns)


# В основном в тестовой выборке отсутствуют значения типа "параметры продукта", и в меньшем количестве (4 штуки) значения типа "расчетные характеристики". Отсутствие признаков можно обьяснить тем, что они замеряются и/или рассчитываются значительно позже.

# В тестовой выборки в том числе отсутствуют целевые признаки rougher.output.recovery и final.output.recovery, которые будут нужны для расчета метрики качества. Добавим их в тестовую выборку, используя метод merge, в качестве индексов используем колонку date

# In[69]:


data_test_0 = data_test
data_test = data_test.merge(data_full.loc[:, ['date','rougher.output.recovery','final.output.recovery']], on='date')
data_test.shape


# In[70]:


df = data_full.loc[:, ['date','rougher.output.recovery','final.output.recovery']].set_index('date')
data_test_1 = data_test_0.set_index('date').join(df)
#data_test_1 = data_test_0.merge(df, on = 'date')


# In[71]:


data_test_1.shape


# ### Проведите предобработку данных.

# In[72]:


data_train['date'] = pd.to_datetime(data_train['date'], format='%Y-%m-%d %H:%M:%S')
data_train.dtypes


# In[73]:


data_test['date'] = pd.to_datetime(data_test['date'], format='%Y-%m-%d %H:%M:%S')
data_test.dtypes


# In[74]:


data_full['date'] = pd.to_datetime(data_full['date'], format='%Y-%m-%d %H:%M:%S')
data_full.dtypes


# Перевёл столбец 'date' в формат datetime

# Открыл фреймы, данных посмотрел общую информацию о фреймах, посмотрел данные которые отсутствуют в тестовой выборе, отсутствие данных в тестовой выборке можно обьяснить тем что они добовляются и расчитываются позже чем остальные данные. Добавил в тестовую выборку целевые признаки. Заполнил отсутствуещие значения с помощью функции ffill. Проверил на явные дубликаты их необнаружил. Проверил эффективность обогащения значение оказалось небольшим.

# ## Анализ данных

# ### Посмотрим, как меняется концентрация металлов (Au, Ag, Pb) на различных этапах очистки.

# In[75]:


plt.figure(figsize=[15, 5])
sns.histplot(data_train['rougher.input.feed_au'], color='y', label='Сырье')
sns.histplot(data_train['rougher.output.concentrate_au'], color='b', label='Концентрация после флотации')
sns.histplot(data_train['primary_cleaner.output.concentrate_au'], color='g', label='Концентрация после первичной очистки')
sns.histplot(data_train['final.output.concentrate_au'], color='r', label='Финальная концетрация металла')
plt.title('График концентрации золота на различных этапах очистки')
plt.xlabel('Концентрация золота')
plt.ylabel('Количество наблюдений')
plt.legend()
plt.show()


# Доля золота на после каждого этапа очитски постепенно увеличивается.

# In[76]:


plt.figure(figsize=[15, 5])
sns.histplot(data_train['rougher.input.feed_ag'], color='y', label='Сырье')
sns.histplot(data_train['rougher.output.concentrate_ag'], color='b', label='Концентрация после флотации')
sns.histplot(data_train['primary_cleaner.output.concentrate_ag'], color='g', label='Концентрация после первичной очистки')
sns.histplot(data_train['final.output.concentrate_ag'], color='r', label='Финальная концетрация серебра')
plt.title('График концентрации серебра на различных этапах очистки')
plt.xlabel('Концентрация серебра')
plt.ylabel('Количество наблюдений')
plt.legend()
plt.show()


# Доля серебра увеличивается на этапе флотации, а затем постепенно уменьшается.

# In[77]:


plt.figure(figsize=[15, 5])
sns.histplot(data_train['rougher.input.feed_pb'], color='y', label='Сырье')
sns.histplot(data_train['rougher.output.concentrate_pb'], color='b', label='Концентрация после флотации')
sns.histplot(data_train['primary_cleaner.output.concentrate_pb'], color='g', label='Концентрация после первичной очистки')
sns.histplot(data_train['final.output.concentrate_pb'], color='r', label='Финальная концетрация свинца')
plt.title('График концентрации свинца на различных этапах очистки')
plt.xlabel('Концентрация свинца')
plt.ylabel('Количество наблюдений')
plt.legend()
plt.show()


# Доля свинца сначала увеличивается на этапе флотации и после первичной очитски, затем остатется примерно на одном уровне.
# 
# В данных для каждого металла присутствуют аномалии(нулевые значения).

# ### Сравним распределение размеров гранул сырья на обучающей и тестовой выборках.

# In[78]:


plt.figure(figsize=[15, 7])
plt.hist(data_train['rougher.input.feed_size'], bins=100, density = True, label='Обучающая выборка', color='y')
plt.hist(data_test['rougher.input.feed_size'], bins =100, density = True, label='Тестовая выборка', alpha=0.8)
plt.xlabel('Размер гранул')
plt.ylabel('Значения')
plt.legend()
plt.show()


# Для этапа флотации распределение гранул находится примерно в одинаковом диапазоне

# In[79]:


plt.figure(figsize=[15, 7])
plt.hist(data_train['primary_cleaner.input.feed_size'], bins=100, density = True, label='Обучающая выборка', color='y')
plt.hist(data_test['primary_cleaner.input.feed_size'], bins =100, density = True, label='Тестовая выборка', alpha=0.8)
plt.xlabel('Размер гранул')
plt.ylabel('Значения')
plt.legend()
plt.show()


# ### Исследуем суммарную концентрацию всех веществ на разных стадиях.

# In[80]:


raw_material_input = data_full['rougher.input.feed_au']+data_full['rougher.input.feed_ag']+data_full['rougher.input.feed_pb']+data_full['rougher.input.feed_sol']
raw_material_concentrate = data_full['rougher.output.concentrate_au']+data_full['rougher.output.concentrate_ag']+data_full['rougher.output.concentrate_pb']+data_full['rougher.output.concentrate_sol']
raw_material_final = data_full['final.output.concentrate_au']+data_full['final.output.concentrate_ag']+data_full['final.output.concentrate_pb']+data_full['final.output.concentrate_sol']


plt.figure(figsize=[15, 5])
sns.histplot(raw_material_input, color='y', label='Сырье')
sns.histplot(raw_material_concentrate, color='b', label='Концентрация после флотации')
sns.histplot(raw_material_final, color='g', label='Финальная концентрация')
plt.title('График концентрации всех веществ на разных этапах')
plt.xlabel('Концентрация')
plt.ylabel('Количество наблюдений')
plt.legend()
plt.show()


# Суммарная концентрация веществ увеличивается к финальному этапу, а диапазон распределения суммарной концентрации веществ уменьшается.

# In[81]:


data_train = data_train[(data_train['rougher.output.concentrate_au'] != 0) & (data_train['rougher.output.concentrate_ag'] != 0) & (data_train['rougher.output.concentrate_pb'] != 0) & (data_train['rougher.output.concentrate_sol'] != 0)]
data_train = data_train[(data_train['final.output.concentrate_au'] != 0) & (data_train['final.output.concentrate_ag'] != 0) & (data_train['final.output.concentrate_pb'] != 0) & (data_train['final.output.concentrate_sol'] != 0)]


# Для финального этапа и этапа флотации для суммарной концентрации веществ присутствуют аномалии ( значения в районе 0). Посчитал необходимым удалить аномалии из данных т.к. аномалии могут негативно повлиять на качество обучения модели

# Концентрация золота увеличивается с каждым этапом очистки, а серебра сначало увеличиваетса а потом уменьшается. Концентрация свинца увеличивается после первого этапа, а за тем остаётся примерно на одном уровне, скорее всего это происходит из за того что удельный вес золота и свинца примерно равный. Суммарная концентрация веществ увеличивается с каждым этапом очистки, это происходит из за удаления пустой породы во время очистки 

# ## Модель

# In[82]:


for meaning in ['rougher.output.recovery','final.output.recovery']:
    missing_columns.remove(meaning)

print('Количество признаков для удаления', len(missing_columns))

data_train = data_train.drop(columns=missing_columns, axis=1)


# Удалил отсутствуещие в тестовой выборке признаки за исключением целевых из обучающей выборки

# ### Напишим функцию для вычисления итоговой sMAPE.

# In[83]:


def smape(target, prediction):
    mape = abs(target - prediction)/((abs(target) + abs(prediction))/ 2)
    smape = (1/len(target))*(mape[:len(target)].sum())*100
    return smape
    
scorer = make_scorer(smape, greater_is_better=False)


# In[84]:


def final_smape(rougher, final):
    final = 0.25*rougher+0.75*final
    return final


# ### Обучим разные модели и оценим их качество кросс-валидацией. Выберем лучшую модель и проверим её на тестовой выборке.

# Подготовим данные для обчения моделей

# In[85]:


features_train = data_train.drop(['rougher.output.recovery','final.output.recovery', 'date'], axis=1)
target_rougher_train = data_train['rougher.output.recovery']
target_final_train = data_train['final.output.recovery']

STATE = RandomState(12345)


# ###### Модель LinearRegression

# In[86]:


get_ipython().run_cell_magic('time', '', 'model_regression = linear_model.LinearRegression()\nscore_rougher = cross_val_score(model_regression, features_train, target_rougher_train, scoring=scorer, cv=100)\nprint(score_rougher.mean()*-1)\n\nmodel_regression_fi = linear_model.LinearRegression()\nscore_final = cross_val_score(model_regression_fi, features_train, target_final_train, scoring=scorer, cv=100)\n\nprint(score_final.mean()*-1)')


# ###### Модель DecisionTreeRegressor

# In[92]:


param_grid = [{'max_depth': [3, 5, 7], 'ccp_alpha': [0.2, 0.4, 0.5, 0.8]},]

model_tree = DecisionTreeRegressor()
grid_search = GridSearchCV(model_tree, param_grid, cv=100, scoring=scorer)

grid_search.fit(features_train, target_rougher_train)


# In[93]:


grid_search.best_params_


# In[94]:


grid_search.best_estimator_


# In[32]:


get_ipython().run_cell_magic('time', '', "for i in range(1, 10):\n    for n in range(5, 6):\n        model_tree = DecisionTreeRegressor(max_depth=n, ccp_alpha=i/10,  random_state=STATE)\n        score_rougher = cross_val_score(model_tree, features_train, target_rougher_train,  scoring=scorer, cv=100)\n        print('max_depth = ', n, 'ccp_alpha=', i/10, 'sMAPE = ', score_rougher.mean()*-1)\n\nprint()\n\nfor i in range(1, 10):\n    for n in range(4, 5):\n        model_tree_fi = DecisionTreeRegressor(max_depth=n, ccp_alpha=i/10, random_state=STATE)\n        score_final = cross_val_score(model_tree_fi, features_train, target_final_train, scoring=scorer, cv=100)\n        print('max_depth = ', n, 'ccp_alpha=', i/10, 'sMAPE = ', score_final.mean()*-1)   ")


# на модели model_tree лучший показатель max_depth = 5, ccp_alpha= 0.5, c значением sMAPE = 5.583172899660774
# 
# на модели model_tree_fi лучший показатель mmax_depth = 4, ccp_alpha= 0.5, с значением sMAPE = 8.81791948731815

# ###### Модель RandomForestRegressor

# In[69]:


get_ipython().run_cell_magic('time', '', "for n in range(31, 32, 10):\n    model_forest = RandomForestRegressor(random_state=state, n_estimators=81, max_depth=n, n_jobs=-1)\n    score_rougher = cross_val_score(model_forest, features_train, target_rougher_train, scoring=scorer, cv=100)\n    print('max_depth = ', n, score_rougher.mean()*-1)")


# n_estimators =  81, max_depth = 31, score_rougher = 4.623761417846823

# In[70]:


get_ipython().run_cell_magic('time', '', "for n in range(11, 12):\n    model_forest_fi = RandomForestRegressor(random_state=state, n_estimators=51, max_depth=11, n_jobs=-1)\n    score_final = cross_val_score(model_forest_fi, features_train, target_final_train, scoring=scorer, cv=100)\n    print('max_depth = ', n, score_final.mean()*-1)")


# n_estimators = 51, max_depth =  11, score_final = 8.203120703619142

# ###### Модель LASSO

# In[33]:


scaler = StandardScaler()
features = scaler.fit_transform(features_train)


# In[34]:


get_ipython().run_cell_magic('time', '', "for n in range(1, 10):\n    model_lasso = linear_model.Lasso(alpha=n/10)\n    score_rougher = cross_val_score(model_lasso, features, target_rougher_train, scoring=scorer, cv=100)\n    print('alpha = ', n/10, score_rougher.mean()*-1)")


# alpha = 0.1, score_rougher = 5.246395174082923

# In[35]:


get_ipython().run_cell_magic('time', '', "for n in range(1, 10):\n    model_lasso_fi = linear_model.Lasso(alpha=n/10)\n    score_final = cross_val_score(model_lasso_fi, features, target_final_train, scoring=scorer, cv=100)\n    print('alpha = ', n/10, score_final.mean()*-1)")


# alpha = 0.2, score_final = 8.701761225459077

# Лучшие результаты показала модель случайного леса с параметрами n_estimators = 81, max_depth = 31, для целевого признака 'rougher.output.recovery', и парамтрами n_estimators = 51, max_depth = 11 для целевого признака 'final.output.recovery'.
# Эти модели и проверим на тестовой выборке

# In[95]:


features_test = data_test.drop(['rougher.output.recovery','final.output.recovery','date'], axis=1)
target_rougher_test = data_test['rougher.output.recovery']
target_final_test = data_test['final.output.recovery']


# подготовил тестовые данные

# In[96]:


get_ipython().run_cell_magic('time', '', "model_forest_test = RandomForestRegressor(random_state=STATE, n_estimators=81, max_depth=31, n_jobs=-1).fit(features_train, target_rougher_train)\npredicted_rougher = model_forest_test.predict(features_test)\nsmape_rougher_test = smape(target_rougher_test, predicted_rougher)\n\nmodel_forest_test_final = RandomForestRegressor(random_state=STATE, n_estimators=51, max_depth=11, n_jobs=-1).fit(features_train, target_final_train)\npredicted_final = model_forest_test_final.predict(features_test)\nsmape_final_test = smape(target_final_test, predicted_final)\n\nsmape_final = final_smape(smape_rougher_test, smape_final_test)\n\nprint('Итоговый SMAPE для тестовой выборки составляет -', smape_final)")


# Полученый результат итогового SMAPE для тестовой выборки невысок и составляет составляет - 9.222955896627479 значит модель работает правельно

# ###### Проверим модель на адекватность с помощью DummyRegressor

# In[100]:


get_ipython().run_cell_magic('time', '', 'dummy_regr = DummyRegressor(strategy="mean").fit(features_train, target_rougher_train)\npredicted_rougher_dummy = dummy_regr.predict(features_test)\nsmape_rougher_test_dummy = smape(target_rougher_test, predicted_rougher_dummy)\n\ndummy_regr_final = DummyRegressor(strategy="mean").fit(features_train, target_final_train)\npredicted_final_dummy = dummy_regr_final.predict(features_test)\nsmape_final_test_dummy = smape(target_final_test, predicted_final_dummy)\n\nsmape_final = final_smape(smape_rougher_test_dummy, smape_final_test_dummy)\n\nprint(\'Итоговый SMAPE для константной модели составляет - \', smape_final)')


# Мне нужно было подготовить прототип модели машинного обучения для «Цифры». Компания разрабатывает решения для эффективной работы промышленных предприятий.
# 
# Модель должна предсказать коэффициент восстановления золота из золотосодержащей руды. Используйте данные с параметрами добычи и очистки.
# 
# Модель поможет оптимизировать производство, чтобы не запускать предприятие с убыточными характеристиками.
# Для выполнения работы было предоставлено 3 фрейма данных(со всеми данными, обучающая выборка и тестовая выборка).
# Изучив данные я увидел пропуски в данных заполнил их ближайшими значениями, проверил на дубликаты их необнаружил.
# Проанализировав столбцы отсутствуещие в тестовой выборке пришёл к выводу что эти данные отсутствуют по пречине того что они заполняются на поздних этапах производства.
# Добавил в тестовую выборку целевые признаки из общего фрейма с данными.
# Для лчшего обучения моделей удалил из обучающей выборки столбцы с данными которые отсутствуют в тестовой выборке.
# Опробовал несколько моделей лучший результат показала модель случайного леса с разными пораметрами для каждого из целевых признаков её я и опробовал на тестовой выборке.
# На тестовой выборке модель показала результат итогового SMAPE = 9.253623965947245. Полученый результат итогового SMAPE для тестовой выборки невысок значит модель работает нормально.
# Проверил модель на адекватность с помощью модели DummyRegressor. Модель DummyRegressor показала результат итогового SMAPE =  19.031795724142526 что на много больше чем у тестовой выборки. Считаю что модель прошла проверку на адекватность.
