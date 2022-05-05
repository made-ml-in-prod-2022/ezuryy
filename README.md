# Домашнее задание 1 по курсу Машинное обучение в продакшене

Пример использования

Установка: 
~~~
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~

Обучение модели:
~~~
python ml_project/train.py
~~~

Создание предикта:
~~~
python ml_project/predict.py
~~~


Требуется сделать "production ready" проект для решения задачи классификации, то есть написать код для обучения и предикта, покрыть его тестами и тд.

Для обучения модели использован небольшой датасет для классификации https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci.

**Критерии (указаны максимальные баллы, по каждому критерию ревьюер может поставить баллы частично):**

- [ ] В описании к пулл реквесту описаны основные "архитектурные" и тактические решения, которые сделаны в вашей работе. В общем, описание того, что именно вы сделали и для чего, чтобы вашим ревьюерам было легче понять ваш код (1 балл)
- [ ] В пулл-реквесте проведена самооценка, распишите по каждому пункту выполнен ли критерий или нет и на сколько баллов(частично или полностью) (1 балл)

- [x] Выполнено EDA, закоммитьте ноутбук в папку с ноутбуками (1 балл)
   Вы так же можете построить в ноутбуке прототип(если это вписывается в ваш стиль работы)

   Можете использовать не ноутбук, а скрипт, который сгенерит отчет, закоммитьте и скрипт и отчет (за это + 1 балл)

- [x] Написана функция/класс для тренировки модели, вызов оформлен как утилита командной строки, записана в readme инструкцию по запуску (3 балла)
- [x] Написана функция/класс predict (вызов оформлен как утилита командной строки), которая примет на вход артефакт/ы от обучения, тестовую выборку (без меток) и запишет предикт по заданному пути, инструкция по вызову записана в readme (3 балла)

- [ ] Проект имеет модульную структуру (2 балла)
- [ ] Использованы логгеры (2 балла)

- [ ] Написаны тесты на отдельные модули и на прогон обучения и predict (3 балла)

- [ ] Для тестов генерируются синтетические данные, приближенные к реальным (2 балла)
   - можно посмотреть на библиотеки https://faker.readthedocs.io/en/, https://feature-forge.readthedocs.io/en/latest/
   - можно просто руками посоздавать данных, собственноручно написанными функциями.

- [ ] Обучение модели конфигурируется с помощью конфигов в json или yaml, закоммитьте как минимум 2 корректные конфигурации, с помощью которых можно обучить модель (разные модели, стратегии split, preprocessing) (3 балла)
- [x] Используются датаклассы для сущностей из конфига, а не голые dict (2 балла)

- [ ] Напишите кастомный трансформер и протестируйте его (3 балла)
   https://towardsdatascience.com/pipelines-custom-transformers-in-scikit-learn-the-step-by-step-guide-with-python-code-4a7d9b068156

- [x] В проекте зафиксированы все зависимости (1 балл)
- [ ] Настроен CI для прогона тестов, линтера на основе github actions (3 балла).
Пример с пары: https://github.com/demo-ml-cicd/ml-python-package
- [ ] Используется hydra  (https://hydra.cc/docs/intro/) (3 балла - доп баллы)
