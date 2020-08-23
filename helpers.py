import matplotlib.pyplot as plt

import os
import pickle
import copy
import datetime
import traceback
import random

import pandas as pd
import numpy as np

from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV

from xgboost import XGBClassifier
from xgboost import plot_importance

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.autograd import Variable
from torch.utils.data import DataLoader


def f1_scorer(y_labels, y_pred_proba):
    return f1_score(y_labels, np.round(y_pred_proba))


def load_dataset(filename='pickled_clicks.beh', pages_threshold=2):
    """
   Функция загрузки данных из сеализованных в pickle tuples
   :param path: путь к директории data
   :param filename: имя файла с данными
   :param pages_threshold: значение порога переходов по страницам.
            Посетитель и его переходы игнорируются, если количество переходов меньше значения параметра.
   :return: кортеж из 5 элементов:
       - идентификаторы посетителей
       - метки классов (1 - бизнес, 0 - не бизнес)
       - количество переходов по URL каждого посетителя
       - незакодированные URLs
       - закодированные URLs
    """
    with open(os.getcwd() + '\\data\\' + filename, 'rb') as data_file:
        data = pickle.load(data_file)
        user_ids, business_labels, navigation_counts, decoded_visits, encoded_visits = list(zip(*data))

    selector = np.array(navigation_counts) >= pages_threshold

    business_labels = np.array(business_labels)[selector]
    encoded_visits = np.array(encoded_visits)[selector]
    decoded_visits = np.array(decoded_visits)[selector]
    navigation_counts = np.array(navigation_counts)[selector]
    user_ids = np.array(user_ids)[selector]

    return user_ids, business_labels, navigation_counts, decoded_visits, encoded_visits


def init_random_seed(value=0):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)
    torch.backends.cudnn.deterministic = True


def prepare_train_test_data(vectorizer, encoded_visits, business_labels):
    """
      Разбивка на тренировочный и тестовый датасеты
    """
    doc_term_matrix = vectorizer.fit_transform(encoded_visits)

    doc_term_matrix_df = pd.DataFrame.sparse.from_spmatrix(doc_term_matrix)
    doc_term_matrix_df.columns = vectorizer.get_feature_names()

    train_data, test_data, train_labels, test_labels = train_test_split(doc_term_matrix_df, business_labels,
                                                                        test_size=0.33, random_state=0)

    return train_data.values, test_data.values, train_labels, test_labels


def store_classification_results(train_labels, test_labels, train_predicted_labels, test_predicted_labels,
                                 algo_name, classification_results):
    """
        Сохраняет результаты классфификации в специальную переменную для дальнейшего сравнения.
    """
    log_loss_train = log_loss(train_labels, train_predicted_labels)
    # print Loss and classification metrics for train data
    print('TRAIN DATASET:')
    # print('Logloss value:', log_loss_train)
    print(classification_report(train_labels, train_predicted_labels, digits=3))

    log_loss_test = log_loss(test_labels, test_predicted_labels)
    # print Loss and classification metrics for test data
    print('TEST DATASET:')
    # print('Logloss value:', log_loss_test)
    print(classification_report(test_labels, test_predicted_labels, digits=3))

    train_report_dict = classification_report(train_labels, train_predicted_labels, digits=3, output_dict=True)
    test_report_dict = classification_report(test_labels, test_predicted_labels, digits=3, output_dict=True)

    classification_results[algo_name] = (train_report_dict, test_report_dict, log_loss_train, log_loss_test)


def fit_predict_logistic_reg(train_data, test_data, train_labels, test_labels, classification_results, algo_name):
    """
        Обучает логистическую регрессию, предсказывает результаты на тренировочном и тестовом датасетах.
    """
    model = LogisticRegressionCV(
        random_state=0,
        tol=1e-3,
        solver='liblinear',
        verbose=1,
        n_jobs=4,
        scoring='f1'
    )

    model.fit(train_data, train_labels)

    store_classification_results(
        train_labels,
        test_labels,
        model.predict(train_data),
        model.predict(test_data),
        algo_name,
        classification_results
    )

    return model


def f1_eval(y_pred, dtrain):
    return 'f1_err', 1 - f1_score(dtrain.get_label(), np.round(y_pred))


def fit_predict_xgboost(
        train_data, test_data, train_labels, test_labels, classification_results, algo_name,
        early_stopping_rounds=20):
    """
        Обучает градиентный бустинг, предсказывает результаты на тренировочном и тестовом датасетах.
    """
    model = XGBClassifier(n_estimators=100, n_jobs=4, random_state=0, max_depth=6)

    model.fit(
        train_data,
        train_labels,
        eval_metric=f1_eval,
        eval_set=[(test_data, test_labels)],
        early_stopping_rounds=early_stopping_rounds,
        verbose=True
    )

    store_classification_results(
        train_labels,
        test_labels,
        model.predict(train_data),
        model.predict(test_data),
        algo_name,
        classification_results
    )

    return model


def plot_importance_xgboost(model):
    """
        Отображает 10 самых важных features с точки срения XGBoost классификатора
    """
    plot_importance(model, max_num_features=10)
    plt.show()


def show_importance_scores(feature_importances, feature_names, topn=10):
    """
        Отображает 10 самых важных features и их названия в порядке убывания важности
    """
    indices = (feature_importances > 0).nonzero()[0]
    adjusted_urls = np.array(feature_names)[feature_importances > 0]
    importance_scores = feature_importances[feature_importances > 0]

    # show top 10 features descending by importance
    importance_df = pd.DataFrame(
        list(zip(indices, adjusted_urls, importance_scores)),
        columns=['Index', 'Adjusted URL', 'Importance Score'])
    importance_df = importance_df.sort_values(by=['Importance Score'], ascending=False).head(topn)

    return importance_df


def get_serialized_models_scores(pickle_filepath='./bus_class_scores_df.sc'):
    if os.path.isfile(pickle_filepath):
        pickled_bus_class_scores_df = pd.read_pickle(pickle_filepath)
        return pickled_bus_class_scores_df


def serialize_models_scores(classification_results, pickle_filepath='./bus_class_scores_df.sc'):
    """
        Сереиализует результаты классфикации в файл в виде pandas dataframe
    """
    bus_class_scores = {
        'Classifier': [key for key in classification_results],
        'F1 test': [value[1]['1']['f1-score'] for key, value in classification_results.items()],
        'Precision test': [value[1]['1']['precision'] for key, value in classification_results.items()],
        'Recall test': [value[1]['1']['recall'] for key, value in classification_results.items()],
        'Logloss test': [value[3] for key, value in classification_results.items()],
        'F1 train': [value[0]['1']['f1-score'] for key, value in classification_results.items()],
        'Precision train': [value[0]['1']['precision'] for key, value in
                            classification_results.items()],
        'Recall train': [value[0]['1']['recall'] for key, value in classification_results.items()],
        'Logloss train': [value[2] for key, value in classification_results.items()]
    }

    bus_class_scores_df = pd.DataFrame(bus_class_scores)

    if os.path.isfile(pickle_filepath):
        pickled_bus_class_scores_df = pd.read_pickle(pickle_filepath)
        bus_class_scores_df = pd.concat(
            [bus_class_scores_df, pickled_bus_class_scores_df]).drop_duplicates().reset_index(drop=True)

    bus_class_scores_df.to_pickle(pickle_filepath)

    return bus_class_scores_df


class VisitorsDataset(Dataset):
    """
        Определяет dataset посетителей для PyTorch
    """

    def __init__(self, visits, labels, token_to_id, max_len, pad_token):
        self.visits = self.to_matrix(visits, token_to_id, max_len, pad_token)
        self.labels = labels

    def __len__(self):
        return self.visits.shape[0]

    def __getitem__(self, i):
        visit = torch.from_numpy(self.visits[i]).long()
        label = torch.tensor(self.labels[i], dtype=torch.long)
        return visit, label

    @staticmethod
    def to_matrix(data, token_to_id, max_len, pad_token):
        data_s = [visit.split() for visit in data]

        max_len = max_len or max(map(len, data_s))
        data_ix = np.zeros([len(data), max_len], dtype='int32') + token_to_id[pad_token]

        for i in range(len(data_s)):
            line_ix = [token_to_id[token] for token in data_s[i]]
            data_ix[i, (max_len - len(line_ix)):max_len] = line_ix

        return data_ix


class RNNLoop(nn.Module):
    def __init__(self, num_tokens, emb_size=256, rnn_num_units=32):
        super(self.__class__, self).__init__()
        self.emb = nn.Embedding(num_tokens, emb_size)
        self.rnn = nn.RNN(emb_size, rnn_num_units, batch_first=True)
        self.hid_to_logit = nn.Linear(rnn_num_units, 1)

    def forward(self, x):
        _, h_last = self.rnn(self.emb(x))
        next_logit = self.hid_to_logit(h_last)
        return next_logit

    def inference(self, x):
        x = self.forward(x)
        x = torch.sigmoid(x)
        return x


def copy_data_to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return [copy_data_to_device(elem, device) for elem in data]
    raise ValueError('Недопустимый тип данных {}'.format(type(data)))


def train_eval_loop(model, train_dataset, val_dataset, criterion, scorer=None,
                    lr=1e-4, epoch_n=10, batch_size=32,
                    device=None, early_stopping_patience=10, l2_reg_alpha=0,
                    max_batches_per_epoch_train=10000,
                    max_batches_per_epoch_val=1000,
                    data_loader_ctor=DataLoader,
                    optimizer_ctor=None,
                    lr_scheduler_ctor=None,
                    shuffle_train=True,
                    dataloader_workers_n=0):
    """
    Цикл для обучения модели. После каждой эпохи качество модели оценивается по отложенной выборке.
    :param model: torch.nn.Module - обучаемая модель
    :param train_dataset: torch.utils.data.Dataset - данные для обучения
    :param val_dataset: torch.utils.data.Dataset - данные для оценки качества
    :param criterion: функция потерь для настройки модели
    :param scorer: функция скоринга для оценки модели на валидации
    :param lr: скорость обучения
    :param epoch_n: максимальное количество эпох
    :param batch_size: количество примеров, обрабатываемых моделью за одну итерацию
    :param device: cuda/cpu - устройство, на котором выполнять вычисления
    :param early_stopping_patience: наибольшее количество эпох, в течение которых допускается
        отсутствие улучшения модели, чтобы обучение продолжалось.
    :param l2_reg_alpha: коэффициент L2-регуляризации
    :param max_batches_per_epoch_train: максимальное количество итераций на одну эпоху обучения
    :param max_batches_per_epoch_val: максимальное количество итераций на одну эпоху валидации
    :param data_loader_ctor: функция для создания объекта, преобразующего датасет в батчи
        (по умолчанию torch.utils.data.DataLoader)
    :return: кортеж из двух элементов:
        - среднее значение функции потерь на валидации на лучшей эпохе
        - лучшая модель
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    model.to(device)

    if optimizer_ctor is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg_alpha)
    else:
        optimizer = optimizer_ctor(model.parameters(), lr=lr)

    if lr_scheduler_ctor is not None:
        lr_scheduler = lr_scheduler_ctor(optimizer)
    else:
        lr_scheduler = None

    train_dataloader = data_loader_ctor(train_dataset, batch_size=batch_size, shuffle=shuffle_train,
                                        num_workers=dataloader_workers_n)
    val_dataloader = data_loader_ctor(val_dataset, batch_size=batch_size, shuffle=False,
                                      num_workers=dataloader_workers_n)

    best_val_loss = float('inf')
    best_val_score = 0.
    best_epoch_i = 0
    best_model = copy.deepcopy(model)

    for epoch_i in range(epoch_n):
        try:
            epoch_start = datetime.datetime.now()
            print('Эпоха {}'.format(epoch_i))

            model.train()
            sum_train_loss = 0
            train_batches_n = 0
            np_train_pred = []
            np_train_y = []

            for batch_i, (batch_x, batch_y) in enumerate(train_dataloader):
                if batch_i > max_batches_per_epoch_train:
                    break

                batch_x = copy_data_to_device(batch_x, device)
                batch_y = copy_data_to_device(batch_y, device)

                pred = model(batch_x)
                loss = criterion(pred, batch_y.view_as(pred).to(torch.float32))

                if scorer is not None:
                    np_train_pred = np.append(np_train_pred,
                                              torch.sigmoid(pred).view_as(batch_y).cpu().detach().numpy().flatten())
                    np_train_y = np.append(np_train_y, batch_y.cpu().detach().numpy().flatten())

                model.zero_grad()
                loss.backward()

                optimizer.step()

                sum_train_loss += float(loss)
                train_batches_n += 1

            print('Эпоха: {} итераций, {:0.2f} сек'.format(train_batches_n,
                                                           (datetime.datetime.now() - epoch_start).total_seconds()))
            print('Суммарное значение функции потерь на обучении', sum_train_loss)
            if scorer is not None:
                train_score = scorer(np_train_y, np_train_pred)
                print('{0} оценка на обучении'.format(scorer.__name__), train_score)

            model.eval()
            sum_val_loss = 0
            val_batches_n = 0
            np_val_pred = []
            np_val_y = []

            with torch.no_grad():
                for batch_i, (batch_x, batch_y) in enumerate(val_dataloader):
                    if batch_i > max_batches_per_epoch_val:
                        break

                    batch_x = copy_data_to_device(batch_x, device)
                    batch_y = copy_data_to_device(batch_y, device)

                    pred = model(batch_x)
                    loss = criterion(pred, batch_y.view_as(pred).to(torch.float32))

                    if scorer is not None:
                        np_val_pred = np.append(np_val_pred,
                                                torch.sigmoid(pred).view_as(batch_y).cpu().detach().numpy().flatten())
                        np_val_y = np.append(np_val_y, batch_y.cpu().detach().numpy().flatten())

                    sum_val_loss += float(loss)
                    val_batches_n += 1

            print('Суммарное значение функции потерь на валидации', sum_val_loss)
            if scorer is not None:
                val_score = scorer(np_val_y, np_val_pred)
                print('{0} оценка на валидации'.format(scorer.__name__), val_score)

            if scorer is not None:
                if val_score > best_val_score:
                    best_epoch_i = epoch_i
                    best_val_score = val_score
                    best_val_loss = sum_val_loss
                    best_model = copy.deepcopy(model)
                    print('Новая лучшая модель!')
                elif epoch_i - best_epoch_i > early_stopping_patience:
                    print('Модель не улучшилась за последние {} эпох, прекращаем обучение'.format(
                        early_stopping_patience))
                    break
            else:
                if sum_val_loss < best_val_loss:
                    best_epoch_i = epoch_i
                    best_val_loss = sum_val_loss
                    best_model = copy.deepcopy(model)
                    print('Новая лучшая модель!')
                elif epoch_i - best_epoch_i > early_stopping_patience:
                    print('Модель не улучшилась за последние {} эпох, прекращаем обучение'.format(
                        early_stopping_patience))
                    break

            if lr_scheduler is not None:
                lr_scheduler.step(sum_val_loss)

            print()
        except KeyboardInterrupt:
            print('Досрочно остановлено пользователем')
            break
        except Exception as ex:
            print('Ошибка при обучении: {}\n{}'.format(ex, traceback.format_exc()))
            break

    return best_val_score, best_val_loss, best_model


def rnn_predict_and_store_results(model, train_dataset, test_dataset, device, classification_results, algo_name):
    with torch.no_grad():
        x_train = copy_data_to_device(train_dataset[0:len(train_dataset)][0], device)
        y_train = train_dataset[0:len(train_dataset)][1].detach().numpy().flatten()
        y_train_pred = model.inference(x_train).cpu().detach().numpy().flatten()

        x_test = copy_data_to_device(test_dataset[0:len(test_dataset)][0], device)
        y_test = test_dataset[0:len(test_dataset)][1].detach().numpy().flatten()
        y_test_pred = model.inference(x_test).cpu().detach().numpy().flatten()

        store_classification_results(
            y_train,
            y_test,
            np.round(y_train_pred),
            np.round(y_test_pred),
            algo_name,
            classification_results
        )


def save_one_hot_test_dataset(test_data, test_labels):
    pickle.dump(zip(test_data, test_labels), open(os.getcwd() + '\\data\\one_hot.ds', 'wb'))


def load_one_hot_test_dataset():
    return pickle.load(open(os.getcwd() + '\\data\\one_hot.ds', 'rb'))


def save_rnn_test_dataset(test_dataset):
    # pickle.dump(zip(test_data, test_labels), open(os.getcwd() + '\\data\\rnn.ds', 'wb'))
    torch.save(test_dataset, os.getcwd() + '\\data\\rnn.ds')


def load_rnn_test_dataset():
    # return pickle.load(open(os.getcwd() + '\\data\\rnn.ds', 'rb'))
    return torch.load(os.getcwd() + '\\data\\rnn.ds')


def save_model(model, model_name):
    pickle.dump(model, open(os.getcwd() + '\\models\\{0}.mdl'.format(model_name), 'wb'))


def load_model(model_name):
    return pickle.load(open(os.getcwd() + '\\models\\{0}.mdl'.format(model_name), 'rb'))
