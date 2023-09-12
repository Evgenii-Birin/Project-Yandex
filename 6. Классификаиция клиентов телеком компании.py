#!/usr/bin/env python
# coding: utf-8

# # Рекомендация тарифов

# В вашем распоряжении данные о поведении клиентов, которые уже перешли на эти тарифы (из проекта курса «Статистический анализ данных»). Нужно построить модель для задачи классификации, которая выберет подходящий тариф. Предобработка данных не понадобится — вы её уже сделали.
# 
# Постройте модель с максимально большим значением *accuracy*. Чтобы сдать проект успешно, нужно довести долю правильных ответов по крайней мере до 0.75. Проверьте *accuracy* на тестовой выборке самостоятельно.

# ## Откройте и изучите файл

# In[19]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier


# Импортировал библиотеку пандас и библиотеки для разбивки фрейма с данными и создания моделей

# In[20]:


df = pd.read_csv('/datasets/users_behavior.csv')
df.info()
df.head()
df.isna().sum()


# импортировал библиотеку Pandas, открыл файл с данными, вывел первичную информацию, и вывел первые 5 строк
# так же проверил данные на наличие пропусков данных в столбцах, пропусков не обнаружил, во всех столбцах находятся только числовые значения

# ## Разбейте данные на выборки

# In[21]:


df_v, df_train = train_test_split(df, test_size=0.8, random_state=12345)
df_valid, df_test = train_test_split(df_v, test_size=0.5, random_state=12345)
display(df_train)
display(df_valid)
display(df_test)


# разбил фрейм с данными на 3 выборки

# ## Исследуйте модели

# выясняем наилучший показатель максимальной глубины дерева решений

# In[22]:


features_train = df_train.drop(['is_ultra'], axis=1)
target_train = df_train['is_ultra']
features_valid = df_valid.drop(['is_ultra'], axis=1)
target_valid = df_valid['is_ultra']

for depth in range(1, 12):
    model_1 = DecisionTreeClassifier(random_state=12345, max_depth=depth, max_features=4, criterion='gini')

    model_1.fit(features_train, target_train)

    predictions_valid = model_1.predict(features_valid)

    print("max_depth =", depth, ": ", end='')
    print(accuracy_score(target_valid, predictions_valid))


# лучшее значение max_depth = 10 и 11 и значении max_features = 4, если применить параметр criterion='gini' результат не меняется, при значении criterion='entropy' результат хуже

# Инициализация модели случайного леса

# In[23]:


best_model = None
best_result = 0
for est in range(1, 210):
    model_2 = RandomForestClassifier(random_state=12345, n_estimators=est, criterion='entropy', n_jobs=-1)
    model_2.fit(features_train, target_train)
    result = model_2.score(features_valid, target_valid)
    if result > best_result:
        best_model = model_2
        best_result = model_2.score(features_valid, target_valid)
        best_est = est

print("Accuracy наилучшей модели на валидационной выборке:", best_result, 'Наилучшее значение n_estimators =', best_est)


# наилучший показатель на модели случайного леса 0.83 при значении n_estimators= 34 и criterion='entropy'

# Инициализация модели логистической регрессии

# In[24]:




model_3 = LogisticRegression(random_state=12345, solver='liblinear', max_iter=1000, penalty='l1')
model_3.fit(features_train, target_train)
model_3.predict(features_valid)
result = model_3.score(features_valid, target_valid)

print('Accuracy логистической регрессии', result)


# Лучший результат показала модель случайного леса с показателем n_estimators = 34
# Худший показатель у модели логистической регрессии 
# Проверять тестовую выборку решил на модели случайного леса с показателем n_estimators = 34 и параметре criterion='entropy'

# ## Проверьте модель на тестовой выборке

# In[25]:


features_test = df_test.drop(['is_ultra'], axis=1)
target_test = df_test['is_ultra']
result = best_model.score(features_test, target_test)

print('Результат Accuracy по тестовой выборке равен:', result)


# На тестовой выборке показатель Accuracy снизился и стался равен 0.79 

# ## (бонус) Проверьте модели на адекватность

# In[33]:


from sklearn.dummy import DummyClassifier


dummy_1 = DummyClassifier(strategy = 'most_frequent', random_state=12345)
dummy_1.fit(features_train, target_train)
result_1 = dummy_1.score(features_test, target_test)

dummy_2 = DummyClassifier(random_state=12345, strategy = 'prior')
dummy_2.fit(features_train, target_train)
dummy_2.predict(features_valid)
result_2 = dummy_2.score(features_test, target_test)

dummy_3 = DummyClassifier(random_state=12345, strategy = 'stratified')
dummy_3.fit(features_train, target_train)
result_3 = dummy_3.score(features_test, target_test)

dummy_4 = DummyClassifier(random_state=12345, strategy = 'uniform')
dummy_4.fit(features_test, target_test)
result_4 = dummy_4.score(features_valid, target_valid)

print('Accuracy теста 1', result_1, 'Accuracy теста 2', result_2, 'Accuracy теста 3', result_3, 'Accuracy теста 4', result_4, sep='\n')


# В yашем распоряжении данные о поведении клиентов, которые уже перешли на эти тарифы. Нужно построить модель для задачи классификации, которая выберет подходящий тариф.
# 
# в ходе выполнения проэкта я начал с того что загрузил нужные мне библиотеки.
# Открыл фрейм данных и сделал базовый просмотр данных проверил на пропуски и определил что столбцы фрейма содержат только цифровые значения.
# 
# разделил данные на 3 выборки в соотношении 80/10/10 выборку в 80% я использовал для обучения моделей одну выборку в 10% использовал как валидационную, третью выборку оставил как тестовую и использовал в конце для проверки модели которая покажет себя лучше других
# 
# на этапе иследования модуля машинного обучения я использовал 3 алгоритма машинного обучения: Дерево, Случайный лес, Логистическая регрессия.
# на этом этапе лучше всего себя показала модель Случайного леса
# 
# на этапе проверки я использовал модуль Случайного леса как показавшего лучший результат, на тестовой выборке результат оказался немного хуже чем на валидационной
# 
# на этапе проверки на адекватность попробова использовать модель DummyClassifier с разными стратегиями, Accuracy тест по всем стратегиям оказался низким 
# 
