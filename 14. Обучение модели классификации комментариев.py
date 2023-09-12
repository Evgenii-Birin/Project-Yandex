#!/usr/bin/env python
# coding: utf-8

# <h1>Содержание<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Подготовка" data-toc-modified-id="Подготовка-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Подготовка</a></span></li><li><span><a href="#Обучение" data-toc-modified-id="Обучение-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Обучение</a></span></li><li><span><a href="#Выводы" data-toc-modified-id="Выводы-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Выводы</a></span></li><li><span><a href="#Чек-лист-проверки" data-toc-modified-id="Чек-лист-проверки-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Чек-лист проверки</a></span></li></ul></div>

# # Проект для «Викишоп»

# Интернет-магазин «Викишоп» запускает новый сервис. Теперь пользователи могут редактировать и дополнять описания товаров, как в вики-сообществах. То есть клиенты предлагают свои правки и комментируют изменения других. Магазину нужен инструмент, который будет искать токсичные комментарии и отправлять их на модерацию. 
# 
# Обучите модель классифицировать комментарии на позитивные и негативные. В вашем распоряжении набор данных с разметкой о токсичности правок.
# 
# Постройте модель со значением метрики качества *F1* не меньше 0.75. 
# 
# **Инструкция по выполнению проекта**
# 
# 1. Загрузите и подготовьте данные.
# 2. Обучите разные модели. 
# 3. Сделайте выводы.
# 
# Для выполнения проекта применять *BERT* необязательно, но вы можете попробовать.
# 
# **Описание данных**
# 
# Данные находятся в файле `toxic_comments.csv`. Столбец *text* в нём содержит текст комментария, а *toxic* — целевой признак.

# ## Подготовка

# In[1]:


import numpy as np 
import pandas as pd
import re
import spacy

import torch 
import transformers 
from tqdm import notebook

import nltk
from nltk.corpus import stopwords as nltk_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.dummy import DummyClassifier


# In[2]:


df = pd.read_csv('/datasets/toxic_comments.csv')
df = df.sample(50000).reset_index(drop=True)


# Загрузил фрейм с данными

# In[3]:


df.info()


# In[4]:


df.head()


# Вывел первые пять строк фрейма. Похоже столбец Unnamed это копия индекса удалим его.

# In[5]:


df = df.drop('Unnamed: 0', axis=1)
df.head()


# In[6]:


text_lemm = []

def lower_sub(text):
    lower_sub = ' '.join(re.sub(r'[^a-zA-Z]', ' ', text).lower().split())
    print(lower_sub)
    print()
    return lower_sub


# In[7]:


df['lower_sub'] = df['text'].apply(lower_sub)


# In[8]:


df.head()


# In[9]:


nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
#nlp.pipe(batch_size=1000)
def lemmatize(text_lemm):
    nlp.pipe(text_lemm, batch_size=1000)
    doc = nlp(text_lemm)
    lemmatize = ' '.join([token.lemma_ for token in doc])
    print(lemmatize)
    print()
    return lemmatize


# In[10]:


df['text_lemm'] = df['lower_sub'].apply(lemmatize)


# In[11]:


df.head()


# In[12]:


features = df['text_lemm']
target = df['toxic']

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.3, shuffle=False)


# Выделил признаки и целевой признак, разделил данные на обучающую и тестовые выборки

# In[13]:


features_train.shape


# In[14]:


corpus = features_train.values.astype('U')
nltk.download('stopwords')
stopwords = set(nltk_stopwords.words('english'))
stopword = ['a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', "aren't",
            'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'couldn',
            "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down',
            'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have',
            'haven', "haven't", 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if',
            'in', 'into', 'is', 'isn', "isn't", 'it', "it's", 'its', 'itself', 'just', 'll', 'm', 'ma', 'me', 'mightn',
            "mightn't", 'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 
            'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're',
            's', 'same', 'shan', "shan't", 'she', "she's", 'should', "should've", 'shouldn', "shouldn't", 'so', 'some', 'such',
            't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they',
            'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't", 'we',
            'were', 'weren', "weren't", 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with',
            'won', "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours',
            'yourself', 'yourselves'
           ]


# In[15]:


tv = TfidfVectorizer()
tf_idf =tv.fit_transform(corpus)


# In[16]:


corpus_test = features_test.values.astype('U')
tf_idf_test = tv.transform(corpus_test)


# ## Обучение

# ###  LogisticRegression

# In[18]:


model = LogisticRegression(random_state=12345)
parametrs = {'C': np.linspace(2, 10, 3),'class_weight':[None, 'balanced']}
model_1 = GridSearchCV(model, parametrs, scoring='f1', cv=3, n_jobs=-1)
model_1.fit(tf_idf, target_train)

#predictions_log = model.predict(tf_idf)
#f_1_log = f1_score(target_train, predictions_log)
#display(f_1_log)


# In[19]:


model_1.best_score_


# In[21]:


model_1.best_params_


# <div class="alert alert-danger">
# <b>Комментарий ревьюера ❌:</b> Необходимо воспользоваться кросс валидацией, иначе таким образом мы получаем слишком позитивный результат и невозможно нормально оценивать метрику.
# 
# Воспользуйся автоматическим методом подбора параметров GridSearchCV (подберем лучшие параметры используя кросс валидацию)
# 
# </div>

# <div class="alert alert-info"> <b>Комментарий студента:</b> Сделал через крос валидацию.</div>

# ### DecisionTreeRegressor

# In[22]:


classifier = DecisionTreeClassifier(random_state=12345)
parametrs_1 = {'max_depth': [100, 150, 200],'class_weight':[None, 'balanced']}
classifier_1 = GridSearchCV(classifier, parametrs_1, scoring='f1', cv=3, n_jobs=-1)

classifier_1.fit(tf_idf, target_train)

#predictions_classifier = classifier.predict(tf_idf)
#f_1_classifier = f1_score(target_train, predictions_classifier)
#print(f_1_classifier)
classifier_1.best_score_


# In[24]:


classifier_1.best_params_


# ### RandomForestClassifier

# In[25]:


clf = RandomForestClassifier(random_state=12345)

parametrs = {'n_estimators': range (10, 51, 10), 'max_depth': range (1,13, 2), 'class_weight':[None, 'balanced']}
grid = GridSearchCV(clf, parametrs, scoring='f1', cv=3, n_jobs=-1)
grid.fit(tf_idf, target_train)

#f_1_grid = f1_score(target_train, predictions_grid)
#print(f_1_grid)
grid.best_score_


# In[26]:


grid.best_params_


# ## Выводы

# Лучше всего себя показала модель LogisticRegression её и проверим на тестовой выборе

# In[28]:


predictions_log_test = model_1.predict(tf_idf_test)
f_1_log_test = f1_score(target_test, predictions_log_test)
print(f_1_log_test)


# <div class="alert alert-success">
# <b>Комментарий ревьюера ✔️:</b> 
# 
# Хорошо:)
# </div>

# In[29]:


dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(tf_idf, target_train)
pred = dummy_clf.predict(tf_idf_test)
f_1_dummy_test = f1_score(target_test, pred)
print(f_1_dummy_test)


# Модель прошла проверку на адекватность

# <div class="alert alert-success">
# <b>Комментарий ревьюера ✔️:</b> 
# 
# Для работы с текстами используют и другие подходы. Например, сейчас активно используются RNN (LSTM) и трансформеры (BERT и другие с улицы Сезам, например, ELMO). НО! Они не являются панацеей, не всегда они нужны, так как и TF-IDF или Word2Vec + модели из классического ML тоже могут справляться. \
# BERT тяжелый, существует много его вариаций для разных задач, есть готовые модели, есть надстройки над библиотекой transformers. Если, обучать BERT на GPU (можно в Google Colab или Kaggle), то должно быть побыстрее.\
# https://huggingface.co/transformers/model_doc/bert.html \
# https://t.me/renat_alimbekov \
# https://colah.github.io/posts/2015-08-Understanding-LSTMs/ - Про LSTM \
# https://web.stanford.edu/~jurafsky/slp3/10.pdf - про энкодер-декодер модели, этеншены\
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html - официальный гайд
# по трансформеру от создателей pytorch\
# https://transformer.huggingface.co/ - поболтать с трансформером \
# Библиотеки: allennlp, fairseq, transformers, tensorflow-text — множество реализованных
# методов для трансформеров методов NLP \
# Word2Vec https://radimrehurek.com/gensim/models/word2vec.html 
#     
# </div>

# ## Чек-лист проверки

# - [x]  Jupyter Notebook открыт
# - [x]  Весь код выполняется без ошибок
# - [x]  Ячейки с кодом расположены в порядке исполнения
# - [x]  Данные загружены и подготовлены
# - [x]  Модели обучены
# - [x]  Значение метрики *F1* не меньше 0.75
# - [x]  Выводы написаны

# <font color='blue'><b>Итоговый комментарий ревьюера</b></font>
# <div class="alert alert-success">
# <b>Комментарий ревьюера ✔️:</b>Евгений, получился хороший проект! 
#     
# Если есть  если есть какие либо вопросы я с удовольствием на них отвечу:) <br> Исправь, пожалуйста, замечания и жду проект на следующую проверку:) </div>
# 
