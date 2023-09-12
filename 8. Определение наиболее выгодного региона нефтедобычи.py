#!/usr/bin/env python
# coding: utf-8

# # Выбор локации для скважины

# Допустим, вы работаете в добывающей компании «ГлавРосГосНефть». Нужно решить, где бурить новую скважину.
# 
# Вам предоставлены пробы нефти в трёх регионах: в каждом 10 000 месторождений, где измерили качество нефти и объём её запасов. Постройте модель машинного обучения, которая поможет определить регион, где добыча принесёт наибольшую прибыль. Проанализируйте возможную прибыль и риски техникой *Bootstrap.*
# 
# Шаги для выбора локации:
# 
# - В избранном регионе ищут месторождения, для каждого определяют значения признаков;
# - Строят модель и оценивают объём запасов;
# - Выбирают месторождения с самым высокими оценками значений. Количество месторождений зависит от бюджета компании и стоимости разработки одной скважины;
# - Прибыль равна суммарной прибыли отобранных месторождений.

# ## Загрузка и подготовка данных

# In[1]:


import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from numpy.random import RandomState
from sklearn import preprocessing


# Импортирую библиотеки

# In[2]:


data_0 = pd.read_csv('/datasets/geo_data_0.csv')
data_1 = pd.read_csv('/datasets/geo_data_1.csv')
data_2 = pd.read_csv('/datasets/geo_data_2.csv')


# Загружаю фреймы данных

# In[3]:


data_0.info()
data_1.info()
data_2.info()


# Вывожу информацию о фреймах данных. Все три фрейма одинаковы по размерам и неимеют пропусков в данных.

# In[4]:


display(data_0.head(), data_1.head(), data_2.head())


# Вывожу первые пять строк каждой таблицы

# In[5]:


display(data_0.duplicated().sum())
display(data_1.duplicated().sum())
display(data_2.duplicated().sum())


# Проверил на явные дубликаты, их не обнаружено

# In[6]:


display(data_0.corr(), data_1.corr(), data_2.corr())


# Признаки друг с другом не коррелируют

# Загрузил фреймы данных. Проверил пропуски в данных и дубликаты, таковых нет.Проверил признаки на корреляцию, признаки друг с другом не коррелируют. Считаю что подготовка данных больше не требуется

# ## Обучение и проверка модели

# In[7]:


TEST_SIZE = 0.25
random = 12345
def on_model(data):
    target = data['product']
    features = data.drop(['id', 'product'], axis=1)
    features = preprocessing.scale(features)
    target_train, target_valid, features_train, features_valid = train_test_split(
        target, features, test_size=TEST_SIZE, random_state=random)
    
    model = LinearRegression()
    model.fit(features_train, target_train)
    
    predicted_valid = model.predict(features_valid)
    rmse = mean_squared_error(target_valid, predicted_valid, squared=False)
    
    predicted_valid_mean = predicted_valid.sum() / len(predicted_valid)
    
    return target_valid, predicted_valid, predicted_valid_mean, rmse
    
target_valid_0, predicted_valid_0, predicted_valid_mean_0, rmse_0 = on_model(data_0)
target_valid_1, predicted_valid_1, predicted_valid_mean_1, rmse_1 = on_model(data_1)
target_valid_2, predicted_valid_2, predicted_valid_mean_2, rmse_2 = on_model(data_2)

print('Регион 1. Средний запас сырья:', predicted_valid_mean_0, 'rmse:', rmse_0)
print('Регион 2. Средний запас сырья:', predicted_valid_mean_1, 'rmse:', rmse_1)
print('Регион 3. Средний запас сырья:', predicted_valid_mean_2, 'rmse:', rmse_2)


# Средний запас сырья в первом и третьем регионе больше чем во втором, судя по данным rmse в первом и третьем регионе большой разброс в запасах сырья, а во втором регионе запасы сырья более равномерны.

# ## Подготовка к расчёту прибыли

# In[8]:


budget_rub = 10*10**9 # Бюджет на разработку 200 скважин в регионе 
total_quantity = 500 # При разведке региона исследуют 500 точек
selected_quantity = 200 # с помощью машинного обучения выбирают 200 точек лучших для разработки
budget_quantity_rub = budget_rub / selected_quantity
barrel_income_rub = 450 # доход с каждого барреля
product_units_income_rub = 450 * 10**3 # доход с каждой единицы продукта
loss_probability_threshold_persent = 2,5


# In[9]:


min_volume_materials = budget_rub / selected_quantity / barrel_income_rub
print(
    'Достаточный объём сырья для безубыточной разработки новой скважины',
    min_volume_materials, 'барреля или', min_volume_materials/(10**3), 'тыс. баррелей')


# Для безубыточной разработки одной сважины требуется содержания не меньше 111111.11111111111 барреля невти в ней

# In[10]:


print(predicted_valid_mean_0 - min_volume_materials/(10**3))
print(predicted_valid_mean_1 - min_volume_materials/(10**3))
print(predicted_valid_mean_2 - min_volume_materials/(10**3))


# Средний запас сырья в скважине во всех трёх регионах ниже чем минимальный расчёт безубыточной разработки скважыны, по этому для выбора региона мы применим машинное оучение
# 

# ## Расчёт прибыли и рисков 

# In[11]:


def series_type(target, pred):
    target = target.reset_index(drop=True)
    pred = pd.Series(pred)
    return target, pred

target_valid_0, predicted_valid_0 = series_type(target_valid_0, predicted_valid_0)
target_valid_1, predicted_valid_1 = series_type(target_valid_1, predicted_valid_1)
target_valid_2, predicted_valid_2 = series_type(target_valid_2, predicted_valid_2)


# Привёл целевые и прогнозные данные к типу Series

# In[12]:


def total_income(target, probabilities, count, product_units_income_rub, budget_rub):
    prob_sort = probabilities.sort_values(ascending=False)
    selected = target[prob_sort.index][:count]
    return int(product_units_income_rub * selected.sum() - budget_rub)


# Фактическая выручка с 200 лучших по прогнозу скважин, минус инвестиции

# In[13]:


def bootstrap_regions(target, predicted, selected_quantity, product_units_income_rub, budget_rub):
    state = RandomState(12345)
    values = []
    for i in range(1000):
        target_subsample = target.sample(n = total_quantity, replace=True, random_state=state)
        pred_subsumple = predicted[target_subsample.index]

        values.append(total_income(target_subsample, pred_subsumple, selected_quantity, product_units_income_rub, budget_rub))
    values = pd.Series(values)
    values_mean = int(values.mean())
    lower = int(values.quantile(q=0.025))
    upper = int(values.quantile(q=0.975))
    risk = len(values[values < 0]) / len(values) * 100
    return values_mean, lower, upper, risk


# Функция получения основных расчётных параметров с помощью будстрепа

# In[14]:


values_mean_0, lower_0, upper_0, risk_0  = bootstrap_regions(
    target_valid_0, predicted_valid_0, selected_quantity, product_units_income_rub, budget_rub)
values_mean_1, lower_1, upper_1, risk_1  = bootstrap_regions(
    target_valid_1, predicted_valid_1, selected_quantity, product_units_income_rub, budget_rub)
values_mean_2, lower_2, upper_2, risk_2  = bootstrap_regions(
    target_valid_2, predicted_valid_2, selected_quantity, product_units_income_rub, budget_rub)


# In[15]:


print('Средняя прибыль лучших месторождений региона 1 равна:', values_mean_0)
print('95% доверительный интервал для средней прибыли 200 лучших месторождений региона 1:', lower_0, ';', upper_0)
print('Риск убытков региона 1 равен:', risk_0, '%')


# In[16]:


print('Средняя прибыль лучших месторождений региона 2 равна:', values_mean_1)
print('95% доверительный интервал для средней прибыли 200 лучших месторождений региона 2:', lower_1, ';', upper_1)
print('Риск убытков региона 2 равен:', risk_1, '%')


# In[17]:


print('Средняя прибыль лучших месторождений региона 3 равна:', values_mean_2)
print('95% доверительный интервал для средней прибыли 200 лучших месторождений региона 3:', lower_2, ';', upper_2)
print('Риск убытков региона 3 равен:', risk_2, '%')


# В данном проекте я провёл загрузку и подготовку данных, обучил модель линейной регрессии, и подготовил прогноз запасов для скважин трёх регионов.
# На основании полученных данных можно сделать вывод что регион номер два (data_1) лучше всего подходит для разработки т.к. у этого региона наибольшая средняя прибыль по лучшим двумста месторождениям, наименьший доверительный интервал и риск убытков составляет 1% (единственный регион удовлетворяющий условиям задачи).
# По полученым данным я бы рекомендовал разработку второго региона(data_1) как наиболее перспективного, хоть и содержащего наименьший запас сырья.
