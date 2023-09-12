#!/usr/bin/env python
# coding: utf-8

# # Задания
# 
#     В самостоятельном проекте вам нужно проанализировать данные о фондах и инвестициях и написать запросы к базе. Задания будут постепенно усложняться, но всё необходимое для их выполнения: операторы, функции, методы работы с базой — вы уже изучили на курсе. К каждому заданию будет небольшая подсказка: она направит вас в нужную сторону, но подробного плана действий не предложит.

# 1. Посчитайте, сколько компаний закрылось

# In[1]:


'''
select count(status)
from company
where status = 'closed';
'''


# 2. Отобразите количество привлечённых средств для новостных компаний США. Используйте данные из таблицы company. Отсортируйте таблицу по убыванию значений в поле funding_total.

# In[2]:


'''
select funding_total
from company
where category_code = 'news'
and country_code = 'USA'
order by funding_total desc;
'''


# 3. Найдите общую сумму сделок по покупке одних компаний другими в долларах. Отберите сделки, которые осуществлялись только за наличные с 2011 по 2013 год включительно.

# In[3]:


'''
select sum(price_amount)
from acquisition
where term_code = 'cash'
and acquired_at > '2010-12-31'
and acquired_at < '2014-01-01';
'''


# 4. Отобразите имя, фамилию и названия аккаунтов людей в твиттере, у которых названия аккаунтов начинаются на 'Silver'.

# In[4]:


'''
select first_name, 
    last_name, 
    twitter_username
from people 
where twitter_username like 'Silver%';
'''


# 5. Выведите на экран всю информацию о людях, у которых названия аккаунтов в твиттере содержат подстроку 'money', а фамилия начинается на 'K'.

# In[5]:


'''
select *
from people
where twitter_username like '%money%' 
and last_name like 'K%';
'''


# 6. Для каждой страны отобразите общую сумму привлечённых инвестиций, которые получили компании, зарегистрированные в этой стране. Страну, в которой зарегистрирована компания, можно определить по коду страны. Отсортируйте данные по убыванию суммы.

# In[6]:


'''
select country_code, 
    sum(funding_total) total
from company
group by country_code
order by total desc;
'''


# 7. Составьте таблицу, в которую войдёт дата проведения раунда, а также минимальное и максимальное значения суммы инвестиций, привлечённых в эту дату. 
#     
#     Оставьте в итоговой таблице только те записи, в которых минимальное значение суммы инвестиций не равно нулю и не равно максимальному значению.

# In[7]:


'''
select cast(funded_at as date) data_fun,
min(raised_amount) min_r,
max(raised_amount) max_r
from funding_round
group by data_fun
having min(raised_amount) > 0 and min(raised_amount) < max(raised_amount)
'''


# 8. Создайте поле с категориями:
#       - Для фондов, которые инвестируют в 100 и более компаний, назначьте категорию high_activity.
#       - Для фондов, которые инвестируют в 20 и более компаний до 100, назначьте категорию middle_activity.
#       - Если количество инвестируемых компаний фонда не достигает 20, назначьте категорию low_activity.
#    
#    Отобразите все поля таблицы fund и новое поле с категориями.

# In[8]:


'''
select *, 
    case 
        when invested_companies > 99 then 'high_activity'
        when invested_companies > 19 then 'middle_activity'
        else 'low_activity'
    end
from fund;
'''


# 9. Для каждой из категорий, назначенных в предыдущем задании, посчитайте округлённое до ближайшего целого числа среднее количество инвестиционных раундов, в которых фонд принимал участие. Выведите на экран категории и среднее число инвестиционных раундов. Отсортируйте таблицу по возрастанию среднего.

# In[9]:


'''
with a as
    (SELECT *, 
           CASE
               WHEN invested_companies>=100 THEN 'high_activity'
               WHEN invested_companies>=20 THEN 'middle_activity'
               ELSE 'low_activity'
           END AS activity
    FROM fund)
select activity,
    round(avg(investment_rounds)) avg_r
from a
group by activity
order by avg_r;
'''


# 10. Проанализируйте, в каких странах находятся фонды, которые чаще всего инвестируют в стартапы. 
# 
#     Для каждой страны посчитайте минимальное, максимальное и среднее число компаний, в которые инвестировали фонды этой страны, основанные с 2010 по 2012 год включительно. Исключите страны с фондами, у которых минимальное число компаний, получивших инвестиции, равно нулю. 
#     
#     Выгрузите десять самых активных стран-инвесторов: отсортируйте таблицу по среднему количеству компаний от большего к меньшему. Затем добавьте сортировку по коду страны в лексикографическом порядке.

# In[10]:


'''
select country_code, 
    min(invested_companies), 
    max(invested_companies), 
    avg(invested_companies)
from fund
where founded_at between '2010-01-01' and '2012-12-31' 
group by country_code
having min(invested_companies) > 0
order by avg(invested_companies) desc, country_code
limit 10;
'''


# 11. Отобразите имя и фамилию всех сотрудников стартапов. Добавьте поле с названием учебного заведения, которое окончил сотрудник, если эта информация известна.

# In[11]:


'''
select a.first_name, 
     a.last_name,
     b.instituition
from people a
left join education b on a.id = b.person_id

'''


# 12. Для каждой компании найдите количество учебных заведений, которые окончили её сотрудники. Выведите название компании и число уникальных названий учебных заведений. Составьте топ-5 компаний по количеству университетов.

# In[12]:


'''
with a as    
    (select distinct c.name, c.id id_com, p.id
    from company c
    join people p on c.id = p.company_id)

select name, count(distinct e.instituition)
from a
join education e on a.id = e.person_id
group by name
order by count(distinct e.instituition) desc
limit 5;
'''


# 13. Составьте список с уникальными названиями закрытых компаний, для которых первый раунд финансирования оказался последним.

# In[13]:


'''
select distinct name
from company
where status = 'closed' and id in (select company_id
from funding_round
where is_first_round = 1 and is_last_round = 1)
'''


# 14. Составьте список уникальных номеров сотрудников, которые работают в компаниях, отобранных в предыдущем задании.

# In[15]:


'''
select distinct id
from people
where company_id in(select distinct id
from company
where status = 'closed' 
and id in (select company_id
           from funding_round
           where is_first_round = 1 
           and is_last_round = 1));
'''


# 16. Посчитайте количество учебных заведений для каждого сотрудника из предыдущего задания. При подсчёте учитывайте, что некоторые сотрудники могли окончить одно и то же заведение дважды.

# In[16]:


'''
select distinct person_id, count(instituition)
from education
where person_id in (select distinct id
from people
where company_id in(select distinct id
from company
where status = 'closed' 
and id in (select company_id
           from funding_round
           where is_first_round = 1 
           and is_last_round = 1)))
group by person_id;
'''


# 17. Дополните предыдущий запрос и выведите среднее число учебных заведений (всех, не только уникальных), которые окончили сотрудники разных компаний. Нужно вывести только одну запись, группировка здесь не понадобится.

# In[17]:


'''
select avg(a.count_in)
from (select distinct person_id, count(instituition) count_in
from education
where person_id in (select distinct id
from people
where company_id in(select distinct id
from company
where status = 'closed' 
and id in (select company_id
           from funding_round
           where is_first_round = 1 
           and is_last_round = 1)))
group by person_id) as a
'''


# 18. Напишите похожий запрос: выведите среднее число учебных заведений (всех, не только уникальных), которые окончили сотрудники Facebook*.
# *(сервис, запрещённый на территории РФ)

# In[19]:


'''
select avg(count_in)
from (select person_id, count(instituition) count_in
from education
where person_id in (select id
    from people
    where company_id in (select id
                        from company
                        where name = 'Facebook'))
group by person_id) as a
'''


# 19. Составьте таблицу из полей:
#     - name_of_fund — название фонда;
#     - name_of_company — название компании;
#     - amount — сумма инвестиций, которую привлекла компания в раунде.
# 
# В таблицу войдут данные о компаниях, в истории которых было больше шести важных этапов, а раунды финансирования проходили с 2012 по 2013 год включительно.

# In[20]:


'''
with a as 
(select *
from investment),
b as 
(select *
from fund),
c as 
(select *
from company),
d as 
(select *
from funding_round)

select b.name name_of_fund,
c.name name_of_company,
d.raised_amount amount
from a
left join c on a.company_id = c.id
left join b on a.fund_id = b.id
left join d on a.funding_round_id = d.id
where c.milestones > 6 
and extract(year from d.funded_at) in (2012, 2013)
'''


# 20. Выгрузите таблицу, в которой будут такие поля:
#     - название компании-покупателя;
#     - сумма сделки;
#     - название компании, которую купили;
#     - сумма инвестиций, вложенных в купленную компанию;
#     - доля, которая отображает, во сколько раз сумма покупки превысила сумму вложенных в компанию инвестиций, округлённая до ближайшего целого числа.
#     
#     Не учитывайте те сделки, в которых сумма покупки равна нулю. Если сумма инвестиций в компанию равна нулю, исключите такую компанию из таблицы. 
#     
#     Отсортируйте таблицу по сумме сделки от большей к меньшей, а затем по названию купленной компании в лексикографическом порядке. Ограничьте таблицу первыми десятью записями.

# In[21]:


'''
select b.name name_ac, --компания покупатель
a.price_amount, -- сумма сделки
c.name name_acq, -- компания которую купили
c.funding_total, -- сумма инвестиций
round(a.price_amount / c.funding_total)
from acquisition a
left join company b on a.acquiring_company_id = b.id
left join company c on a.acquired_company_id = c.id
where a.price_amount != 0 and c.funding_total != 0
order by a.price_amount desc, name_acq
limit 10;
'''


# 21. Выгрузите таблицу, в которую войдут названия компаний из категории social, получившие финансирование с 2010 по 2013 год включительно. Проверьте, что сумма инвестиций не равна нулю. Выведите также номер месяца, в котором проходил раунд финансирования.

# In[22]:


'''
select a.name, extract(month from cast(b.funded_at as date))
from company a
left join funding_round b on a.id = b.company_id
where category_code = 'social' 
and extract(year from cast(b.funded_at as date)) >= '2010'
and extract(year from cast(b.funded_at as date)) <= '2013'
and b.raised_amount > 0
'''


# 22. Отберите данные по месяцам с 2010 по 2013 год, когда проходили инвестиционные раунды. Сгруппируйте данные по номеру месяца и получите таблицу, в которой будут поля:
#     - номер месяца, в котором проходили раунды;
#     - количество уникальных названий фондов из США, которые инвестировали в этом месяце;
#     - количество компаний, купленных за этот месяц;
#     - общая сумма сделок по покупкам в этом месяце.

# In[23]:


'''
with aa as
(select extract(month from cast(a.funded_at as date)) as_date, count(distinct c.name) count_name
from funding_round a 
join investment b on a.id = b.funding_round_id
join fund c on b.fund_id = c.id
where extract(year from cast(a.funded_at as date)) between '2010' and '2013' 
and c.country_code = 'USA'
group by as_date),
ab as (select extract(month from cast(acquired_at as date)) ac_date,
       count(acquired_company_id) count_ac, 
       sum(price_amount) sum_price
      from acquisition
      where extract(year from cast(acquired_at as date)) between '2010' and '2013'
      group by ac_date)
select as_date, count_name, count_ac, sum_price
from aa
left join ab on aa.as_date = ab.ac_date
'''


# 23. Составьте сводную таблицу и выведите среднюю сумму инвестиций для стран, в которых есть стартапы, зарегистрированные в 2011, 2012 и 2013 годах. Данные за каждый год должны быть в отдельном поле. Отсортируйте таблицу по среднему значению инвестиций за 2011 год от большего к меньшему.

# In[24]:


'''
with a as
    (select country_code, avg(funding_total) in_11
    from company
    where extract(year from founded_at) = 2011
    and country_code is not null
    group by 1),
b as 
    (select country_code, avg(funding_total) in_12
    from company
    where extract(year from founded_at) = 2012
    and country_code is not null
    group by 1),
c as 
    (select country_code, avg(funding_total) in_13
    from company
    where extract(year from founded_at) = 2013
    and country_code is not null
    group by 1)
    
select a.country_code, in_11, in_12, in_13
from a
inner join b on a.country_code = b.country_code
inner join c on a.country_code = c.country_code
order by in_11 desc;
'''

