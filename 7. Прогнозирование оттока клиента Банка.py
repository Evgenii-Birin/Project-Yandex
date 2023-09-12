#!/usr/bin/env python
# coding: utf-8

# # Отток клиентов

# Из «Бета-Банка» стали уходить клиенты. Каждый месяц. Немного, но заметно. Банковские маркетологи посчитали: сохранять текущих клиентов дешевле, чем привлекать новых.
# 
# Нужно спрогнозировать, уйдёт клиент из банка в ближайшее время или нет. Предоставлены исторические данные о поведении клиентов и расторжении договоров с банком. 
# 
# Нужно построить модель с предельно большим значением *F1*-меры. Чтобы сдать проект успешно, нужно довести метрику до 0.59. Проверить *F1*-меру на тестовой выборке самостоятельно.
# 
# Дополнительно измеряйть *AUC-ROC*, сравнивать её значение с *F1*-мерой.
# 
# Источник данных: [https://www.kaggle.com/barelydedicated/bank-customer-churn-modeling](https://www.kaggle.com/barelydedicated/bank-customer-churn-modeling)

# <h1>Содержание<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Подготовка-данных" data-toc-modified-id="Подготовка-данных-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Подготовка данных</a></span></li><li><span><a href="#Исследование-задачи" data-toc-modified-id="Исследование-задачи-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Исследование задачи</a></span></li><li><span><a href="#Борьба-с-дисбалансом" data-toc-modified-id="Борьба-с-дисбалансом-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Борьба с дисбалансом</a></span></li><li><span><a href="#Тестирование-модели" data-toc-modified-id="Тестирование-модели-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Тестирование модели</a></span></li><li><span><a href="#Чек-лист-готовности-проекта" data-toc-modified-id="Чек-лист-готовности-проекта-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Чек-лист готовности проекта</a></span></li></ul></div>

# ## Подготовка данных

# In[66]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler  
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, accuracy_score


# In[67]:


data = pd.read_csv('/datasets/Churn.csv')
data.info()


# Признаки:
# CreditScore — кредитный рейтинг
# Geography — страна проживания
# Gender — пол
# Age — возраст
# Tenure — сколько лет человек является клиентом банка
# Balance — баланс на счёте
# NumOfProducts — количество продуктов банка, используемых клиентом
# HasCrCard — наличие кредитной карты
# IsActiveMember — активность клиента
# EstimatedSalary — предполагаемая зарплата
# Целевой признак:
# Exited - факт ухода клиента
# 
# Втаблице есть столбцы которые по моему мнению не нужны для создания модели.
# Это столбцы: RowNumber, CustomerId, Surname.
# Их я удалю.
# 
# Также в столбце Tenure несть пропущеные значения предположу что пропущенные значения это новые клиенты которые являются клиентами банка меньше года, заменю пропущеные значения на 0  и преведу к целочисленому виду.

# In[68]:


del_col = ['RowNumber', 'CustomerId', 'Surname']
data = data.drop(del_col, axis=1)
data['Tenure']=data['Tenure'].fillna(0).astype('int64')
data.info()


# In[69]:


data.head()


# In[70]:


features = data.drop('Exited', axis=1)
target = data['Exited']


# разделил фрейм данных на признаки и целевой признак

# In[71]:


features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345)
features_valid, features_test, target_valid, target_test = train_test_split(
    features_valid, target_valid, test_size=0.5, random_state=12345)


# разделил данные на обучающую, валидационную и тестовую

# In[72]:


features_train['Geography'] = pd.get_dummies(features_train['Geography'], drop_first=True)
features_train['Gender'] = pd.get_dummies(features_train['Gender'], drop_first=True)

features_valid['Geography'] = pd.get_dummies(features_valid['Geography'], drop_first=True)
features_valid['Gender'] = pd.get_dummies(features_valid['Gender'], drop_first=True)

features_test['Geography'] = pd.get_dummies(features_test['Geography'], drop_first=True)
features_test['Gender'] = pd.get_dummies(features_test['Gender'], drop_first=True)


# In[73]:


display(features_train.head())
display(features_valid.head())
display(features_test.head())


# Подготовил признаки Geography и Gender с помощью One-hot-encoding

# Открыл фрейм данных определил признаки и целевой признак, удалил три слолбца с фамилией клиента, индексом строки в данных и уникальным идентификатором клиента так как считаю что эти данные не нужны для обучения модели так как не несут ни какого признака уйдёт ли клиент или нет. Разделил данные на обучающую выборку, проверочную и тестовую выборки. Так же заменил пропущеные значения в данных солбца, сколько лет человек является клиентом банка, на ноль так как считаю что пропуски в данных означает что человек клиент этого банка меньше года. Подготовил признаки Geography и Gender с помощью One-hot-encoding

# ## Исследование задачи

# In[74]:


target.value_counts(normalize=True).plot(kind='bar', title='Соотношение тех, кто остался и тех кто ушел')


# видно дисбаланс в данных оставшихся клиентов больше

# In[75]:


numeric=['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
scaler=StandardScaler()
scaler.fit(features_train[numeric])
features_train[numeric]=scaler.transform(features_train[numeric])
features_valid[numeric]=scaler.transform(features_valid[numeric])
features_test[numeric]=scaler.transform(features_test[numeric])


# Стандартизирую данные

# In[76]:


model = LogisticRegression(random_state=12345, solver='liblinear')
model.fit(features_train, target_train)
predictions_valid = model.predict(features_valid)
print("Accuracy:", accuracy_score(target_valid, predictions_valid))
print("F1:", f1_score(predictions_valid, target_valid))


# модель логистической регресси показывает результат неудовлетворяющий нас

# In[77]:


get_ipython().run_cell_magic('time', '', 'for depth in range(1,100):\n    model_decision = DecisionTreeClassifier(max_depth=depth,random_state=12345)\n    model_decision.fit(features_train,target_train)\n    prediction = model_decision.predict(features_valid)\n    print(\'max_depth:\',depth,\'F1:\',f1_score(target_valid,prediction))\n    print("Accuracy:", accuracy_score(target_valid, predictions_valid))')


# У модели дерева решений лучший показатель F1 = 0.57 у параметра max_depth равного 5 

# In[78]:


get_ipython().run_cell_magic('time', '', "for depth in range(1,50):\n    model_forest = RandomForestClassifier(max_depth=depth, random_state=12345)\n    model_forest.fit(features_train, target_train)\n    prediction = model_forest.predict(features_valid)\n    print('max_depth:',depth,'F1:',f1_score(target_valid, prediction))\n")


# Лучший показатель max_depth=24

# In[79]:


get_ipython().run_cell_magic('time', '', "for est in range(1,200,1):\n    model_forest = RandomForestClassifier(max_depth=24, n_estimators=est, random_state=12345)\n    model_forest.fit(features_train, target_train)\n    prediction = model_forest.predict(features_valid)\n    print('n_estimators:', est,'F1:', f1_score(target_valid, prediction))")


# Лучший показатель n_estimators: 137, при нем F1: 0.6135

# In[80]:


probabilities_forest = model_forest.predict_proba(features_valid)
probabilities_one_valid_forest = probabilities_forest[:, 1]
fpr_forest,tpr_forest,thresholds = roc_curve(target_valid, probabilities_one_valid_forest)

auc_roc_forest = roc_auc_score(target_valid, probabilities_one_valid_forest)

print(auc_roc_forest)


# In[81]:


probabilities_decision = model_decision.predict_proba(features_valid)
probabilities_one_valid_decision = probabilities_decision[:, 1]
fpr_decision,tpr_decision,thresholds = roc_curve(target_valid, probabilities_one_valid_decision)

auc_roc_decision = roc_auc_score(target_valid, probabilities_one_valid_decision)

print(auc_roc_decision)


# In[82]:


probabilities_regression = model.predict_proba(features_valid)
probabilities_one_valid_regression = probabilities_regression[:, 1]
fpr_regression,tpr_regression,thresholds = roc_curve(target_valid, probabilities_one_valid_regression)

auc_roc_regression = roc_auc_score(target_valid, probabilities_one_valid_regression)

print(auc_roc_regression)


# In[83]:


plt.figure()

plt.figure(figsize=(10,7))
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.plot(fpr_forest,tpr_forest)
plt.plot(fpr_decision,tpr_decision)
plt.plot(fpr_regression,tpr_regression)
plt.xlabel('Частота ложноположительных результатов')
plt.ylabel('Истинный положительный показатель')
plt.title('ROC-кривая')
plt.legend(('Случайная модель', 'Случайный лес', 'Дерево решения', 'Логистическая регрессия'),
           loc= 'upper left') 
plt.show()


# Метрика AUC-ROC так же показывает что модель случайного леса показывает лучшие результаты

# Вывел диограмму целевого признака она показала что данные целевого признака не сбалансирована, оставшихся клиентов больше примерно в 4 раза. Создал три модели на данных без балансировки, лучше всего по параметру F1 показала себя модель случайного леса. Метрика AUC-ROC показывает большую площадь под кривой случайного леса

# ## Борьба с дисбалансом

#  Как показала модель логистической регрессии параметры Accuracy и F1 на несбалансированых данных 0.79 и 0.29 соответственно

# попробуем логистическую регрессию с автоматической балансировкой с помощью параметра class_weight='balanced'

# In[84]:


model_1 = LogisticRegression(random_state=12345, solver='liblinear', class_weight='balanced')
model_1.fit(features_train, target_train)
predictions_1 = model_1.predict(features_valid)
print("Accuracy:", accuracy_score(predictions_1, target_valid))
print("F1:", f1_score(predictions_1, target_valid))


# параметры Accuracy и F1 на сбалансированых с помощью автоматической балансировки данных 0.70 и 0.51 соответственно

# попробуем логистическую регрессию с балансировкой добавлением

# In[85]:


def upsample(features, target, repeat):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)
    
    features_upsampled, target_upsampled = shuffle(
    features_upsampled, target_upsampled, random_state=12345)
    
    return features_upsampled, target_upsampled

features_upsampled, target_upsampled = upsample(features_train, target_train, 4) 
model_2 = LogisticRegression(random_state=12345, solver='liblinear')
model_2.fit(features_upsampled,target_upsampled)
predicted_2 = model_2.predict(features_valid)
print("Accuracy:", accuracy_score(predicted_2, target_valid))
print("F1:", f1_score(target_valid, predicted_2))


# Создал функцию для балансировки добавлением, показатели Accuracy и F1 немного похуже чем с автоматической балансировкой

# попробуем логистическую регрессию с балансировкой уменньшения большего класса

# In[86]:


def downsample(features, target, fraction):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_downsampled = pd.concat(
        [features_zeros.sample(frac=fraction, random_state=12345)] + [features_ones])
    target_downsampled = pd.concat(
        [target_zeros.sample(frac=fraction, random_state=12345)] + [target_ones])
    
    features_downsampled, target_downsampled = shuffle(
        features_downsampled, target_downsampled, random_state=12345)
    
    return features_downsampled, target_downsampled
features_downsampled, target_downsampled = downsample(features_train, target_train, 0.25)
model_3 = LogisticRegression(random_state=12345, solver='liblinear')
model_3.fit(features_downsampled,target_downsampled)
predicted_3 = model_3.predict(features_valid)
print("Accuracy:", accuracy_score(predicted_3, target_valid))
print("F1:", f1_score(target_valid, predicted_3))


# создал функцию балансировки с уменьшением большего класса показатели Accuracy и F1 стали такими же как с автоматической балансировкой

# In[87]:


probabilities_1 = model_1.predict_proba(features_valid)
probabilities_one_valid_1 = probabilities_1[:, 1]
fpr_1,tpr_1,thresholds = roc_curve(target_valid, probabilities_one_valid_1)

auc_roc_1 = roc_auc_score(target_valid, probabilities_one_valid_1)

print(auc_roc_1)


# In[88]:


probabilities_2 = model_2.predict_proba(features_valid)
probabilities_one_valid_2 = probabilities_2[:, 1]
fpr_2,tpr_2,thresholds = roc_curve(target_valid, probabilities_one_valid_2)

auc_roc_2 = roc_auc_score(target_valid, probabilities_one_valid_2)

print(auc_roc_2)


# In[89]:


probabilities_3 = model_3.predict_proba(features_valid)
probabilities_one_valid_3 = probabilities_3[:, 1]
fpr_3,tpr_3,thresholds = roc_curve(target_valid, probabilities_one_valid_3)

auc_roc_3 = roc_auc_score(target_valid, probabilities_one_valid_3)

print(auc_roc_3)


# In[90]:


plt.figure()

plt.figure(figsize=(10,7))
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.plot(fpr_1,tpr_1)
plt.plot(fpr_2,tpr_2)
plt.plot(fpr_3,tpr_3)
plt.xlabel('Частота ложноположительных результатов')
plt.ylabel('Истинный положительный показатель')
plt.title('ROC-кривая')
plt.legend(('Случайная модель', 'Автоматическая функция', 'С уменьшением большего класса', 'С увеличением большего класса'),
           loc= 'upper left') 
plt.show()


# График иследования AUC-ROC показал почти одинаковые результаты иследования, чуть лучше только показатели у балансировки добавлением

# С помощью логистической регрессии попробовал 3 способа балансировки данных, с помощью встроеного параметра, с помощью увеличения меньшего признака, и с помощью уменьшения большего признака. Все способы балансировки опробую на модели которая показала лучший результат без балансировки

# ## Тестирование модели

# На несбалансированых данных лучше всего себя показала модель случайного леса с ппарамертрами n_estimators = 137  и max_depth = 24, при нем F1: 0.59, её и попробуем на тестовой выборке со всеми видами балансировки

# In[91]:


get_ipython().run_cell_magic('time', '', '\nmodel_forest_1 = RandomForestClassifier(max_depth=24, n_estimators=137, random_state=12345, criterion=\'entropy\')\nmodel_forest_1.fit(features_downsampled, target_downsampled)\nprediction_1 = model_forest_1.predict(features_valid)\nprint(\'F1:\', f1_score(target_valid, prediction_1))\nprint("Accuracy:", accuracy_score(prediction_1, target_valid))')


# Попробовал модель случайного леса с балансировкой уменьшения большего класса f1 в пределах параметров указаных в задании

# In[92]:


get_ipython().run_cell_magic('time', '', '\nmodel_forest_2 = RandomForestClassifier(max_depth=24, n_estimators=137, random_state=12345, class_weight=\'balanced\', criterion=\'entropy\')\nmodel_forest_2.fit(features_train, target_train)\nprediction_2 = model_forest_2.predict(features_valid)\nprint(\'F1:\', f1_score(target_valid, prediction_2))\nprint("Accuracy:", accuracy_score(prediction_2, target_valid))')


# Попробовал модель случайного леса с балансировкой параметром class_weight='balanced' f1 впределах нужного нам показателя

# In[93]:


get_ipython().run_cell_magic('time', '', '\nmodel_forest_3 = model_forest_1.fit(features_upsampled, target_upsampled)\nprediction_3 = model_forest_3.predict(features_valid)\nprint(\'F1:\', f1_score(target_valid, prediction_3))\nprint("Accuracy:", accuracy_score(prediction_3, target_valid))')


# модель случайного леса с балансировкой добавлением показала результат f1 = 0.62 и самое большое время на выполнение кода

#   Тестирование

# На валидационной выборке лучше всего себя показала модель случайного дерева с балансировкой добавлением, её и опробуем на тестовой выборке

# In[94]:


prediction_test = model_forest_3.predict(features_test)
print('F1:', f1_score(target_test, prediction_test))
print("Accuracy:", accuracy_score(prediction_test, target_test))


# Модель показала нам результат F1 = 0.60, что выше отметки поставленой в условии задачи

# Исследую метрику AUC-ROC

# In[95]:


probabilities_forest_1 = model_forest_1.predict_proba(features_valid)
probabilities_one_valid_forest_1 = probabilities_forest_1[:, 1]
fpr_forest_1,tpr_forest_1,thresholds = roc_curve(target_valid, probabilities_one_valid_forest)

auc_roc_forest_1 = roc_auc_score(target_valid, probabilities_one_valid_forest)

print(auc_roc_forest_1)


# In[96]:


probabilities_forest_2 = model_forest_2.predict_proba(features_valid)
probabilities_one_valid_forest_2 = probabilities_forest_2[:, 1]
fpr_forest_2,tpr_forest_2,thresholds = roc_curve(target_valid, probabilities_one_valid_forest_2)

auc_roc_forest_2 = roc_auc_score(target_valid, probabilities_one_valid_decision)

print(auc_roc_forest_2)


# In[97]:


probabilities_forest_3 = model_forest_3.predict_proba(features_valid)
probabilities_one_valid_forest_3 = probabilities_forest_3[:, 1]
fpr_forest_3,tpr_forest_3,thresholds = roc_curve(target_valid, probabilities_one_valid_forest_3)

auc_roc_forest_3 = roc_auc_score(target_valid, probabilities_one_valid_forest_3)

print(auc_roc_forest_3)


# In[98]:


probabilities_forest_test = model_forest_3.predict_proba(features_test)
probabilities_one_test_forest = probabilities_forest_test[:, 1]
fpr_forest_test,tpr_forest_test,thresholds = roc_curve(target_test, probabilities_one_test_forest)

auc_roc_forest_test = roc_auc_score(target_test, probabilities_one_test_forest)

print(auc_roc_forest_test)


# In[99]:


plt.figure()

plt.figure(figsize=(10,7))
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.plot(fpr_forest_1,tpr_forest_1)
plt.plot(fpr_forest_2,tpr_forest_2)
plt.plot(fpr_forest_3,tpr_forest_3)
plt.plot(fpr_forest_test,tpr_forest_test)
plt.xlabel('Частота ложноположительных результатов')
plt.ylabel('Истинный положительный показатель')
plt.title('ROC-кривая')
plt.legend(('Случайная модель', 'С уменьшением большего класса', 'Автоматическая функция', 'С увеличением большего класса', 'Тестирование модели'),
           loc= 'upper left') 
plt.show()


# По метрике AUC-ROC на тестовая выборка показала результат чуть хуже чем валидационная

# 

# Из «Бета-Банка» стали уходить клиенты. Каждый месяц. Немного, но заметно. Банковские маркетологи посчитали: сохранять текущих клиентов дешевле, чем привлекать новых.
# 
# Нужно спрогнозировать, уйдёт клиент из банка в ближайшее время или нет. Вам предоставлены исторические данные о поведении клиентов и расторжении договоров с банком.
# 
# Я распаковал данные и провёл первичный анализ данных определил что три столбца с данными(RowNumber, CustomerId, Surname) не представляют интерес для дальнейшего иследования и создания моделей, по этому я удалил их. В столбце Tenure (сколько лет человек является клиентом банка) есть пропущеные значения, я предположил что это люди являются клиентами данного банка меньше года поэтому заменил пропущеные данные на ноль. Подготовил данные столбцов страна проживания и пол клиента с помощью One-hot-encoding.
# 
# Разделил данные на выборки построил диаграмму по целевому признаку, диаграмма показала дисбаланс в данных целевого признака число оставшихся клиентов больше примерно в четыре раза. Создал три разных модели, лучший показатель f1 показала модель случайного леса.
# 
# Написал простую модель логистической регрессии на ее основе проверил три способа балансировки. Все три способа балансировки показали почти одинаковый результат опробую их на модели которая показала лучший результат без балансировки.
# 
# Опробовал три модели балансировки на модели случайного леса лучший показатель f1 показала балансировка добавлением на ней и провёл иследование остальных двух моделей и показателя  AUC-ROC диаграмма AUC-ROC показала также что модель случайного леса лучше показала себя, модель дерева была немного похуже.

# In[ ]:




