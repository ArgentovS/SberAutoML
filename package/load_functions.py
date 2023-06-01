import os
import pandas as pd
import datetime as dt
import missingno as msno
import matplotlib.pylab as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from scipy import stats
import multiprocessing
from multiprocessing import Pool, Manager


### Функция загрузки датасета
def file_load(file_name):
    """
    Функция загружает Датасет из файла источника

        Параметры:
            file_name (str): имя файла с данными
        Выходные параметры (DataFrame)

    """
    file_path = '../Data/' + file_name
    print('... загружаем файл с данными датасета ...')
    df = pd.read_csv(file_path)
    file_info(df, file_path)
    print('\n... готовим иллюстрацию заполненности датасета значениями ...')
    msno.matrix(df)  # визуализируем заполнение занчениями датасета

    return df


### Функция определения характеристик файла источника данных
def file_info(df, file_path):
    """
    Функция описывает файл источник для загруженного датасета

        Параметры:
            df (DataFrame): загруженный датасет
            file_path (sgr): отрносительнрый путь к файлу с датасетом
        Выходные параметры (None)

    """
    print(f'Источником данных является файл: {file_path[8:]}'
          f'\nРазмер файла {os.stat(file_path).st_size} байт'
          f'\nВ файле содержится 1 таблица:'
          f'\n  - количество строк: {df.shape[0]}'
          f'\n  - количество столбцов: {df.shape[1]}')

