# Идентификация B2B-пользователей веб-сайта электронной коммерции на основе clickstream

Этот репозиторий представляет собой набор juputer notebooks и вспомогательных файлов, которые демострируют использование техник NLP (one-hot vs embeddings) для классификации clickstreams.

## Инструкция по запуску 

1) Cклонируйте репозиторий курса:

`git clone https://github.com/hail2002/clickstream_analysis.git`

2) Выполните команду:

`pip install -r requirements.txt`

3) Запустите ноутбук:

`ipython notebook`

## Содержание ноутбуков

[1. Introduction](1_intro.ipynb)
* бизнес-метрики 
* альтернативное решение 
* формализация задачи 

[2. Data](2_data.ipynb)
* датасет из Data Warehouse
* признаковое пространство, сведение задачи к классификации, поиск меток классов
* чистка и разметка данных, семплирование

[3. Models](3_models.ipynb)
* преобразование истории визитов в текст, применение NLP (векторизация, словарь, embeddings)
* baseline-классификатор Logistic Regression на "мешке слов"
* градиентный бустинг (XGBoost) на "мешке слов"
* Vanilla RNN на clickstream с помощью PyTorch, классфификатор на embeddings

[4. Stats](4_stats.ipynb)
* статистическая значимость результатов
