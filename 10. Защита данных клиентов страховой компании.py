#!/usr/bin/env python
# coding: utf-8

# # Защита персональных данных клиентов

# Вам нужно защитить данные клиентов страховой компании «Хоть потоп». Разработайте такой метод преобразования данных, чтобы по ним было сложно восстановить персональную информацию. Обоснуйте корректность его работы.
# 
# Нужно защитить данные, чтобы при преобразовании качество моделей машинного обучения не ухудшилось. Подбирать наилучшую модель не требуется.

# <h1>Содержание<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Загрузка-данных" data-toc-modified-id="Загрузка-данных-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Загрузка данных</a></span></li><li><span><a href="#Умножение-матриц" data-toc-modified-id="Умножение-матриц-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Умножение матриц</a></span></li><li><span><a href="#Алгоритм-преобразования" data-toc-modified-id="Алгоритм-преобразования-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Алгоритм преобразования</a></span></li><li><span><a href="#Проверка-алгоритма" data-toc-modified-id="Проверка-алгоритма-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Проверка алгоритма</a></span></li></ul></div>

# ## Загрузка данных

# In[36]:


import pandas as pd
import numpy as np

from numpy import linalg
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn. model_selection import train_test_split


# In[37]:


df = pd.read_csv('/datasets/insurance.csv')


# Загрузил фрейм с данными

# In[38]:


df.info()


# Вывел информацию о датафрейме

# In[39]:


df.head()


# Вывел первые пять строк фрейма

# In[40]:


df.isnull().sum ()


# проверил датафрейм на пропуски в данных, пропуски необнаружил.

# In[41]:


print(f'В датафрейме {df.duplicated().sum()} дубликатов')


# In[42]:


df[df.duplicated()]


# В датафрейме 153 дубликата, но так как нет уникального id клиента нет уверености что это дубликаты, а не клиенты с похожими данными, так что данные оставлю без изменений.

# In[43]:


matrix = df.values


# Сделал матрицу из датафрейма

# In[44]:


matrix[:5][:]


# Вывел первые пять строк матрицы

# In[45]:


df.shape, matrix.shape


# Вывел размеры датафрейма и матрицы, они одинаковы.

# In[46]:


column = df.columns
print(column)


# Сохранил названия столбцов в отдельную переменную.

# In[47]:


features = df.drop('Страховые выплаты', axis=1)
target = df['Страховые выплаты']


# ## Умножение матриц

# ###### Ответьте на вопрос и обоснуйте решение. 
#  Признаки умножают на обратимую матрицу. Изменится ли качество линейной регрессии?

# В этом задании вы можете записывать формулы в *Jupyter Notebook.*
# 
# Чтобы записать формулу внутри текста, окружите её символами доллара \\$; если снаружи —  двойными символами \\$\\$. Эти формулы записываются на языке вёрстки *LaTeX.* 
# 
# Для примера мы записали формулы линейной регрессии. Можете их скопировать и отредактировать, чтобы решить задачу.
# 
# Работать в *LaTeX* необязательно.

# Обозначения:
# 
# - $X$ — матрица признаков (нулевой столбец состоит из единиц)
# 
# - $y$ — вектор целевого признака
# 
# - $P$ — матрица, на которую умножаются признаки
# 
# - $w$ — вектор весов линейной регрессии (нулевой элемент равен сдвигу)

# In[48]:


X = np.concatenate((np.ones((features.shape[0], 1)), features), axis=1)
y = target.values
P = np.random.random(size=(features.shape[1]+1, features.shape[1]+1))
w = np.zeros(features.shape[1])
w0 = target.mean()


# In[49]:


print(P)


# In[50]:


P_1 = np.linalg.inv(P)
np.allclose(P@P_1, np.eye(P.shape[0]))


# проверил матрицу на обратимость

# Предсказания:
# 
# $$
# a = Xw
# $$
# 
# Задача обучения:
# 
# $$
# w = \arg\min_w MSE(Xw, y)
# $$
# 
# Формула обучения:
# 
# $$
# w = (X^T X)^{-1} X^T y
# $$

# Домноженая матрица
# $$
# a' = XP((XP)^T XP)^{-1} (XP)^T y
# $$
# Раскроем скобки
# $$
# a' = XP(P^T X^T XP)^{-1} P^T X^T y
# $$
# 
# $$
# a' = XP(P^T (X^T X) P)^{-1} P^T X^T y
# $$
# 
# Можшо сократить произведение матриц $ (P^T)^{-1}P^T $$ и $$ P(P)^{-1} $, так как их произведение даёт $ E $.
# 
# $$
# a' = X(X^T X)^{-1} X^T y
# $$
# Если 
# $$
# a = X(X^T X)^{-1} X^T y
# $$
# и
# $$
# a' = X(X^T X)^{-1} X^T y
# $$
# Отсюда следует что $ a = a' $, то есть предсказания не изменятся и качество линейной регрессии тоже

# **Ответ:** Качество линейной регрессии не изменится.
# 
# **Обоснование:**  Обратимая матрица - это матрица которая при умножении на обратную матрицу даёт еденичную матрицу.
# тоесть определитель матрицы не равен 0, Тоесть строки и столбцы линейно независимы и не влияют на качество регрессии.
# 
# Ниже опробуем на практике и сравним показатели R2 регрессии матрицы признаков и матрицы признаков умноженой на обратимую

# In[51]:


reg = linear_model.LinearRegression().fit(X,y)
answer = reg.predict(X)

print(r2_score(y, answer))


# In[52]:


Z = X @ P
reg_1 = linear_model.LinearRegression().fit(Z,y)
answer = reg_1.predict(Z)

print(r2_score(y, answer))


# Как мы видим показатели R2 не отличаются

# ## Алгоритм преобразования

# **Алгоритм**
# 
# Создадим функцию которая принемает на вход матрицу признаков, создаёт рандомную матрицу, проверяет есть ли у неё обратная матрица если обратной матрицы нет то процесс поиска и проверки на обратимость повторяются до тех пор пока не найдётся обратимая матрица, после нахождения обратной матрицы матрица признаков умножается на рандомную обратимую матрицу которая и подаётся в модель. 

# **Обоснование**
# 
# таким образом данные данные клиента подвергнуся своеобразному шифрованию

# ## Проверка алгоритма

# In[53]:


features_train, features_test, target_train,  target_test = train_test_split(
    features, target, test_size= 0.4 , random_state= 12345 )


# In[54]:


model_1 = linear_model.LinearRegression().fit(features_train, target_train)
predictions = model_1.predict(features_test)
print(r2_score(target_test, predictions))


# Построил модель линейной регрессии на  данных фрейма

# In[55]:


n = features.shape[1]+1
i = -1
def linalg (n):
    while i < 0:
        P = np.random.random(size=(n))
        P_1 = np.linalg.inv(Z)
        if np.allclose(P@P_1, np.eye(P.shape[0])) != True:
            continue
        else:
            i += 1
    return P


# Создал функцию создания рандомной обратимой матрицы

# In[56]:


features_matrix = X @ P
train_features, test_features, train_target,  test_target = train_test_split(
    features_matrix, target, test_size= 0.4 , random_state= 12345 )


# In[57]:


model_matrix = linear_model.LinearRegression().fit(train_features, train_target)
predictions_matrix = model_matrix.predict(test_features)
print(r2_score(test_target, predictions_matrix))


# Создал модель линейной регрессии на основе данных датафрейма и обратимой матрицы

# Как мы видим показатель R2 одинаковый в обоих случаях равный 0.4237
