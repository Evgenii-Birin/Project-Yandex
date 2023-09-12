#!/usr/bin/env python
# coding: utf-8

# # Определение возраста покупателей

# ## Описание проекта

# Сетевой супермаркет «Хлеб-Соль» внедряет систему компьютерного зрения для обработки фотографий покупателей. Фотофиксация в прикассовой зоне поможет определять возраст клиентов, чтобы:
# Анализировать покупки и предлагать товары, которые могут заинтересовать покупателей этой возрастной группы;
# Контролировать добросовестность кассиров при продаже алкоголя.

# Нужно построить модель, которая по фотографии определит приблизительный возраст человека. В нашем распоряжении набор фотографий людей с указанием возраста.

# ## Исследовательский анализ данных

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError


# In[2]:


labels = pd.read_csv('/datasets/faces/labels.csv')
educational_datagen = ImageDataGenerator(rescale=1./255)
educational_gen_flow = educational_datagen.flow_from_dataframe(
    dataframe=labels,
    directory='/datasets/faces/final_files/',
    x_col='file_name',
    y_col='real_age',
    target_size=(224, 224),
    batch_size=32,
    class_mode='raw',
    seed=12345)

features, target = next(educational_gen_flow)


# In[3]:


fig = plt.figure(figsize=(10,10))
for i in range(15):
    fig.add_subplot(4, 4, i+1)
    plt.imshow(features[i])
    plt.title(f' Возраст {target[i]}')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()


# In[4]:


labels.info()


# In[9]:


labels['real_age'].hist(bins=100)
plt.title('Real age')
plt.xlabel('Возраст')
plt.ylabel('Количество')
plt.show()


# В выборке содержится 7591 фотография людей с указанием их возраста, что является не средним размером выборки. Больше всего фотографий людей примерно от 18 до 30 лет.

# ## Обучение модели

# Перенесите сюда код обучения модели и её результат вывода на экран.
# 
# 
# (Код в этом разделе запускается в отдельном GPU-тренажёре, поэтому оформлен не как ячейка с кодом, а как код в текстовой ячейке)

# ```python
# 
# optimizer = Adam(0.0001)
# 
# def load_train(path): 
#     labels = pd.read_csv(path + 'labels.csv')
#     
#     datagen = ImageDataGenerator(validation_split=0.25, rescale=1/255,
#                                  horizontal_flip=True,  
#                                  width_shift_range=0.2, 
#                                  height_shift_range=0.2)
#     
#     train_data = datagen.flow_from_dataframe(
#         labels,
#         directory=path + 'final_files/',
#         x_col='file_name',
#         y_col='real_age',
#         target_size=(224, 224),
#         batch_size=32,
#         class_mode='raw',
#         subset='training',
#         seed=12345)
#     return train_data
#     
#     
# def load_test(path):    
#     labels = pd.read_csv(path + 'labels.csv')
#     
#     datagen = ImageDataGenerator(validation_split=0.25, rescale=1/255)
#     
#     test_data = datagen.flow_from_dataframe(
#         labels,
#         directory=path + 'final_files/',
#         x_col='file_name',
#         y_col='real_age',
#         target_size=(224, 224),
#         batch_size=32,
#         class_mode='raw',
#         subset='validation',
#         seed=12345)
#     return test_data
#     
#     
# def create_model(input_shape):
#     
#     backbone = ResNet50(input_shape=input_shape, 
#                         weights='imagenet', 
#                         include_top=False)
# 
#     model = Sequential()
#     model.add(backbone)
#     model.add(GlobalAveragePooling2D())
#     model.add(Dense(1, activation='relu'))
#     model.compile(optimizer=optimizer, loss=MeanSquaredError(),
#                   metrics=[MeanAbsoluteError()])
#     
#     return model
#     
#     
# def train_model(model, train_data, test_data, batch_size=None, epochs=7,
#                steps_per_epoch=None, validation_steps=None):
# 
#     model.fit(train_data, 
#               validation_data=test_data,
#               batch_size=batch_size, epochs=epochs,
#               steps_per_epoch=steps_per_epoch,
#               validation_steps=validation_steps,
#               verbose=2, shuffle=True)
# 
#     return model
# 
# '''

# '''python
# 
# 178/178 - 108s - loss: 253.4202 - mean_absolute_error: 11.4994 - val_loss: 968.1015 - val_mean_absolute_error: 26.3513
# 
# Epoch 2/7
# 178/178 - 90s - loss: 94.3660 - mean_absolute_error: 7.3537 - val_loss: 720.8384 - val_mean_absolute_error: 21.9067
# 
# Epoch 3/7
# 178/178 - 93s - loss: 74.2252 - mean_absolute_error: 6.5253 - val_loss: 240.3712 - val_mean_absolute_error: 11.7886
# 
# Epoch 4/7
# 178/178 - 90s - loss: 56.0916 - mean_absolute_error: 5.7225 - val_loss: 172.6996 - val_mean_absolute_error: 9.8518
# 
# Epoch 5/7
# 178/178 - 97s - loss: 48.0590 - mean_absolute_error: 5.2691 - val_loss: 101.1806 - val_mean_absolute_error: 7.5504
# 
# Epoch 6/7
# 178/178 - 97s - loss: 39.8033 - mean_absolute_error: 4.8407 - val_loss: 81.7959 - val_mean_absolute_error: 6.7224
# 
# Epoch 7/7
# 178/178 - 95s - loss: 36.1916 - mean_absolute_error: 4.6384 - val_loss: 79.1163 - val_mean_absolute_error: 6.4184
# 
# WARNING:tensorflow:sample_weight modes were coerced from
#   ...
#     to  
#   ['...']
# 60/60 - 10s - loss: 79.1163 - mean_absolute_error: 6.4184
# Test MAE: 6.4184
# '''

# ## Анализ обученной модели

# Для обучения я использовал модель нейронной сети ResNet50(50 слоёв), т.к. выборка небольшая замораживаю ResNet50 без верхушки, заморозка позволяет избавиться от переобучения и повысить скорость обучения сети.
# Модель обучилась и показала  значение МАЕ = 6.4 на 7 эпохе что соответствует условию задачи и не является слишком большим значением.
# Для улучшения значения МАЕ можно попробовать добавить количество эпох, поменять значение пораметров  steps_per_epoch=len(train_data) и validation_steps = len(test_data) и поизменять параметр оптимизатора Adamю

# ## Чек-лист

# - [x]  Jupyter Notebook открыт
# - [x]  Весь код выполняется без ошибок
# - [x]  Ячейки с кодом расположены в порядке исполнения
# - [x]  Исследовательский анализ данных выполнен
# - [x]  Результаты исследовательского анализа данных перенесены в финальную тетрадь
# - [x]  MAE модели не больше 8
# - [x]  Код обучения модели скопирован в финальную тетрадь
# - [x]  Результат вывода модели на экран перенесён в финальную тетрадь
# - [x]  По итогам обучения модели сделаны выводы

# In[ ]:




