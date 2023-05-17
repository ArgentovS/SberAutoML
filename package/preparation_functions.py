import os
import pandas as pd
import datetime as dt
import missingno as msno
import matplotlib.pylab as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from scipy import stats


### Функция коррекции данных в датасете
def dataset_preparation(df):
    """
    Функция исправляет типы данных, убирает пропуски и дубликаты в данных датасета

        Параметры:
            df (DataFrame): корректируемый датасет
        Выходные параметры (DataFrame)

    """
    print('\nЭтап 1. ... запускаем проверку идентичности типов данных в полях датасета ...')
    noncorrect_columns = checking_type_error(df)
    print('... запускаем коррекцию типов данных в полях датасета ...')
    correct_type(df, noncorrect_columns)

    print('\nЭтап 2. ... запускаем замену некорректного указания типа None ...')
    df = df.replace(['nan'], np.nan)
    print('Все некорректные указания на пустые значения заменены на None.')

    print('\nЭтап 3. ... запускаем изменение типов данных в полях даты и времени ...')
    if 'visit_date' in df.columns:
        df['visit_date'] = df.visit_date.apply(lambda x:
                                               dt.datetime.strptime(x, '%Y-%m-%d').date())
        print("В поле 'visit_date' тип данных изменён на datetime.")
    if 'visit_time' in df.columns:
        df['visit_time'] = df.visit_time.apply(lambda x:
                                               dt.datetime.strptime(x, '%H:%M:%S').time())
        print("В поле 'visit_time' тип данных изменён на datetime.")
    if 'hit_date' in df.columns:
        df['hit_date'] = df.hit_date.apply(lambda x:
                                           dt.datetime.strptime(x, '%Y-%m-%d').date())
        print("В поле 'hit_date' тип данных изменён на datetime.")

    print('\nЭтап 4. ... анализируем пропущенные значения в датасете ...')
    data_set_audit(df)

    if 'visit_date' in df.columns:
        # Список удаляемых колонок
        # (в колонках более 40% пропущенных данных)
        columns_columns_delete = ['utm_keyword', 'device_os', 'device_model']

        # Список колонок, в которых проверется наличие пропусков для удаления строк
        # (в колонках менее 1% пропущенных данных)
        columns_rows_delete = ['utm_source']

        # Список колонок, в которых меняются пропуски на наиболее часто встречающиеся значения
        # (в колонках более 1% и менее 40% пропущенных данных)
        columns_value_top = ['utm_campaign', 'utm_adcontent', 'device_brand']

        # Список колонок, в которых меняются пропуски на наиболее часто встречающиеся значения
        # при этом одновременно учитываются корреляциии атрибута с другим атрибутом
        # (в колонках более 1% и менее 40% пропущенных данных)
        # columns_value_top_correlation = [('geo_country', 'geo_city')]

    #     else:
    #         # Список удаляемых колонок
    #         # (в колонках более 40% пропущенных данных)
    #         columns_columns_delete = ['hit_referer', 'event_value']

    #         # Список колонок, в которых проверется наличие пропусков для удаления строк
    #         # (в колонках менее 1% пропущенных данных)
    #         columns_rows_delete = ['hit_number', 'hit_type', 'hit_page_path',\
    #                        'event_category', 'event_action']

    #         # Список колонок, в которых меняются пропуски на наиболее часто встречающиеся значения
    #         # (в колонках более 1% и менее 40% пропущенных данных)
    #         columns_value_top = ['device_brand', 'event_label']

    # Удаляем пропуски в датасете

    df = clean_columns_rows(df, columns_columns_delete,
                            columns_rows_delete,
                            columns_value_top)

    # Удаляем дубликаты в датасете
    print('\nЭтап 5. ... запускаем удаление дубликатов ...')
    df = df.drop_duplicates()
    print('Дубликаты записей удалены.')
    print('Размер Датасета после удаление дубликатов:', df.shape)

    if 'visit_date' in df.columns:
        print('\nЭтап 6. ... запускаем поиск и удаление аномалий ...')
        df = delete_anomalies(df)
        print('Записи с клиентами, имеющими аномально большое количество визитов - удалены.')
        print(f'Размер Датасета после удаления аномалий: {df.shape}')

        # Агрегируем и визуализируем распределение атрибутов датасета
        agg_changes(df)



    return df


### Функция проверки всех колонок датасета на единство типа данных в колонке
def checking_type_error(df):
    """
    Функция проверяет каждое поле датасета на несовпадение типов данных и выводит поля с указанием количества записей разных типов

        Параметры:
            df (DataFrame): проверяемый датасет
        Выходные параметры:
            columns_noncorrect_types (list): список полей датасета, в каждом из которых найдены разные типы данных

    """
    flagCorrect = True
    columns_noncorrect_types = []
    for elem in df.columns:
        df_type = df[elem].apply(lambda x: type(x)).value_counts().to_frame()
        if len(df_type) > 1:
            columns_noncorrect_types.append(elem)
            if flagCorrect:
                print('Для одного атрибута обнаружены данные разного типа в следующих полях')
                print(' ------------------------------------------------------------ ')
                print('|  НАИМЕНОВАНИЕ ПОЛЯ  |  ТИП ДАННЫХ  |   КОЛИЧЕСТВО ЗАПИСЕЙ  |')
                print(' ------------------------------------------------------------ ')
            flagCorrect = False

            for i in range(len(df_type)):
                print('   ', df_type.columns[0], ' ' * (21 - len(df_type.columns[0])),
                      str(df_type.index[i])[7:-1], ' ' * (21 - len(str(df_type.index[i])[7:-1])),
                      str(df_type[df_type.columns[0]].values[i]))
            print(' ------------------------------------------------------------ ')
    if flagCorrect:
        print('Полей с разными типами данных в одном атрибуте не обнаружено!')

    return columns_noncorrect_types


### Функция изменения типа данных с 'float' на 'str'
def correct_type(df, noncorrect_columns):
    """
    Функция изменяет типы данных на 'str' в заданных полях датасета

        Параметры:
            df (DataFrame): датасет, в котором корректируются типы данных полей
            noncorrect_columns (list): список полей датасета, в которых функция корректирует тип данных
        Выходные параметры (None)

    """
    for elem in noncorrect_columns:
        df[elem] = df[elem].apply(lambda x: str(x))
    print(f'Коррекция типов данных завершена в {len(noncorrect_columns)} полях')
    print('... запущена перепроверка идентичности типов данных ... ')
    checking_type_error(df)


### Функция проверки наличия незаполненных значений в колоноках датасета
def data_set_audit(df):
    """
    Функция проверяет пропуски в Датасете и выводит для каждого поля долю пропусков в % от общего количесвтва записей

        Параметры:
            df (DataFrame): датасет, в котором проверяются пропуски
        Выходные параметры (None)

    """
    print('Размер анализируемого Датасета:', df.shape)
    nan_columns = [(elem, df[elem].isna().sum(), type(df.loc[0, elem]))
                   for elem in df.columns
                   if df[elem].isnull().describe()[1] > 1 or
                   df[elem].isnull().describe()[1] == 1 and
                   df[elem].isnull()[0] == True]
    if len(nan_columns) > 0:
        print('ПРОПУСКИ  В  КОЛОНКАХ  ДАТАСЕТА')
        print('========================================================================')
        print('  Поле                        Пропуски             Тип данных в колонке ')
        print('                           (кол-во,      %)                             ')
        print('------------------------------------------------------------------------')
        for elem in nan_columns:
            print(' ', elem[0],
                  ' ' * (24 - len(elem[0])), elem[1],
                  ' ' * (9 - len(str(elem[1]))), round((elem[1] / len(df) * 100), 2),
                  ' ' * 12, str(elem[2])[7:-1])
        print('========================================================================')
    else:
        print('Пропущенных данных в Датасете не обнаружено!')


### Функция отчистки столбцов и строк от нулевых значений
def clean_columns_rows(df, col_col_df, col_row_df, col_cell_val_top):
    """
    Функция удаляет заданные столбы. Затем функция удаляет строки, в которых есть пропущеные данные

        Параметры:
            df (DataFrame): датасет, в котором удаляются столбы, затем проверяются пропуски и удаляют соответвующие строки
            col_col_df (list): список колонок, которые необходимо удалить.
            col_row_df (list): списко колонок, при наличии пропусках в которых - необходимо удалить строки.
            col_cell_val_top (list): список колонок, в которых необходимо замеить пропуски простым выбором наиболее часто встречающихся значений.
            col_cell_val_top_corr (list(typle)): списко кортежей, в которых указываются родительские и дочернии колонки для подбора наиболее часто встречающихся значений в дочерней колонке с учтом соответствующего значения в родительской колонке.
        Выходные параметры:
            df_clear (DataFrame): датасет, отчищенный от пропусков

    """

    print('... удаляем колонки, в которых более 40% пропущенных данных ...')
    df = df.drop(columns=col_col_df, axis=1)  # Удаление колонки 'hit_time'
    print('После удаления заданных колонок - размер датасета:', df.shape)

    print('... удаляем строки, для которых в колонках менее 1% пропущенных данных ...')
    df = df.dropna(subset=col_row_df, axis=0, how='any')  # Удаление строк, в которых выявлены пропущенные данные
    print(f'Удаление колонок и строк с нулевыми данными - завершено.')

    print('... меняем пропущенные данные на наиболее часто встречающиеся ... ')
    for col in col_cell_val_top:
        nan_indexes = df[df[col].isnull()].index
        df.loc[nan_indexes, col] = df[col].describe().loc['top']

    print('... запущена перепроверка отсутствия нулевых данных ... ')
    data_set_audit(df)

    #     print('\n')
    #     for column_elem in col_cell_val_top_corr:
    #         parent_column, child_column = column_elem[0], column_elem[1]

    #         # 1. Определяем уникальные значения родительской колонки,
    #         #    имеющие пустые значения в дочерней колонке
    #         columns_non_correct = list(df_clear[parent_column].loc[df_clear[child_column].isnull()].unique())

    #         # 2. заполняем пустоты названиями, встречающимеся чаще всего среди значений
    #         #    дочерней колонки для соответвующего значения родительской колонки Датасета

    #         for col_cor in columns_non_correct:

    #             # Массив индексов пустых значений в дочерней колонке Датасета,
    #             # соответсвующих значению родительской колонки Датасета
    #             index_nan_values = df_clear[child_column].\
    #                             loc[df_clear[child_column].isnull()].\
    #                             loc[df_clear[parent_column] == col_cor].index

    #             # Самое часто встречающееся значение в дочерней колонке,
    #             # соответсвущее значению родительской колонки Датасета
    #             top_value = str(df_clear[child_column].\
    #                             loc[df_clear[child_column].notnull()].\
    #                             loc[df_clear[parent_column] == col_cor].\
    #                             describe().top)

    #             if top_value not in [None, 'nan', 'null', 'Nan', 'NaN']:
    #                 df_clear.loc[index_nan_values, child_column] = top_value

    #         print('... в зависимых колонках меняем пропуски на самые частые значения ... ')
    #         print(f'Заменена нулевых данных, для которых выявлены наиболее часто встречающиеся значения - завершена\n')

    return df


### Функция выявления и удаления аномалий в датасете визитов
def delete_anomalies(df):
    """
    Функция определяет количество аномалий в датасете визитов и удаляет их из датасета

        Параметры:
            df (DataFrame): датасет визитов
        Выходные параметры (DataFrame)

    """
    # Вычисление количесвтва визитов каждого клиента по 'client_id'
    visit_count = df.groupby('client_id').agg('count')

    # Вычисление количесвта дней в периоде фиксации данных в датасете
    days_count = (df.visit_date.max() - df.visit_date.min()).days

    # Определение идентификаторов клиентов, у которых в среднем больше
    # одного визита в день
    roboticity_client = list(visit_count. \
                             loc[visit_count['session_id'] >= days_count].index)

    # Исключение клиентов, у которых в среднем больше одного визита в день
    # df = df.query(f'client_id not in {roboticity_client}')
    df = df.loc[~df.client_id.isin(roboticity_client)]

    return df


### Агрегация атрибутов датасета визитов во времени
def agg_changes(df):
    """
    Функция агрегирует данные датасета для формирования визуализации распределения визитов клиентов по разным атрибутам

        Параметры:
            df (DataFrame): агрегируемый датасет
        Выходные параметры (None)

    """

    # Поле для подсчёта уникальных клиентов
    print('\nЭтап 7. ... запускаем подготовку визуализации по п.2.1.4(пп7) ...')
    df['client'] = df['client_id']

    # Словарь агрегируемых атребутов и функций по атрибутам
    ag_dict = {'client_id': 'max', 'visit_date': 'min', 'visit_time': 'min',
               'geo_country': 'max', 'geo_city': 'max',
               'device_category': 'max', 'device_browser': 'max',
               'utm_source': 'max', 'utm_medium': 'max'}
    print('Словарь функций агрегируемых атрибутов - подготовлен.')
    print('... запускаем агрегацию атрибутов (агрегация займёт время)...')

    # Датасет сагрегированных атрибутов визитов
    df_ag = df.groupby('client').agg(ag_dict)
    print('Агрегация завершена.')

    # Агрегация посещений новыми клиентами по месяцам
    df_ag['year-month'] = df_ag['visit_date'].apply(lambda x: dt.datetime.strftime(x, "%Y-%m"))
    ag_ym = df_ag.groupby('year-month').agg({'client_id': 'count'})

    # Агрегация посещений новыми клиентами по дням месяца
    df_ag['dayOFmonth'] = df_ag['visit_date'].apply(lambda x: dt.datetime.strftime(x, "%d"))
    ag_dm = df_ag.groupby('dayOFmonth').agg({'client_id': 'count'})

    # Агрегация посещений новыми клиентами по дням недели
    df_ag['dayOFweek'] = df_ag['visit_date'].apply(lambda x: dt.datetime.strftime(x, "%w"))
    ag_dw = df_ag.groupby('dayOFweek').agg({'client_id': 'count'})

    # Агрегация посещений новыми клиентами по часам суток
    df_ag['hourOFday'] = df_ag['visit_time'].apply(lambda x: x.hour)
    ag_hd = df_ag.groupby('hourOFday').agg({'client_id': 'count'})

    # Агрегация структуры стран нахождения клиентов
    ag_country = df_ag.groupby('geo_country'). \
        agg({'client_id': 'count'}).sort_values('client_id', ascending=False)
    country_count = (ag_country.index.to_list()[:5],
                     ag_country.client_id.to_list()[:5])

    # Агрегация структуры городов нахождения клиентов
    ag_city = df_ag.groupby('geo_city'). \
        agg({'client_id': 'count'}).sort_values('client_id', ascending=False)
    city_count = (ag_city.index.to_list()[:10],
                  ag_city.client_id.to_list()[:10])

    # Агрегация структуры типов дивайсов, которые используют клиенты
    ag_device = df_ag.groupby('device_category'). \
        agg({'client_id': 'count'}).sort_values('client_id', ascending=False)
    divice_count = (ag_device.index.to_list(),
                    ag_device.client_id.to_list())

    # Агрегация структуры браузеров, которые используют клиенты
    ag_browser = df_ag.groupby('device_browser'). \
        agg({'client_id': 'count'}).sort_values('client_id', ascending=False)
    browser_count = (ag_browser.index.to_list()[:10],
                     ag_browser.client_id.to_list()[:10])

    # Агрегация структуры каналов привлечения клиентов
    ag_source = df_ag.groupby('utm_source'). \
        agg({'client_id': 'count'}).sort_values('client_id', ascending=False)
    source_count = (ag_source.index.to_list()[:10],
                    ag_source.client_id.to_list()[:10])

    # Агрегация структуры типов привлечения клиентов
    ag_medium = df_ag.groupby('utm_medium'). \
        agg({'client_id': 'count'}).sort_values('client_id', ascending=False)
    medium_count = (ag_medium.index.to_list()[:10],
                    ag_medium.client_id.to_list()[:10])

    print('Данные для вывода графиков готовы.')

    dfs = [(ag_ym, 'по месяцам'), (ag_dm, 'по дням месяца'),
           (ag_dw, 'по дням недели'), (ag_hd, 'по часам дня'),
           (country_count, 'Страны нахождения клиентов'),
           (city_count, 'Города нахождения клиентов'),
           (divice_count, 'Типы устройств, используемые клиентами'),
           (browser_count, 'Браузеры, используемые клиентами'),
           (source_count, 'Каналы привлечения клиентов'),
           (medium_count, 'Типы привлечения клиентов')]

    time_plot(dfs[:4])
    structure_plot(dfs[4:])


### Формирование полотна частоты посещения сайта
def time_plot(dfs):
    """
    Функция визуализирует распределение визитов клиентов по временным периодам

        Параметры:
            dfs (DataFrame): cагрегированная часть датасета визитов, касающаяся распределения визитов по временным периодами
        Выходные параметры (None)

    """

    print('\nИЗМЕНЕНИЕ ПО ПЕРИОДАМ ВРЕМЕНИ КОЛИЧЕСТВА ВИЗИТОВ КЛИЕНТОВ НА САЙТ')
    fig, ax = plt.subplots(figsize=(15, 3))
    ax.set_title(f'Визиты {dfs[0][1]}')
    ax.plot(dfs[0][0].index, dfs[0][0]['client_id'], color='green')

    fig, ax = plt.subplots(figsize=(15, 3))
    ax.set_title(f'Визиты {dfs[1][1]}')
    ax.plot(dfs[1][0].index, dfs[1][0]['client_id'], color='green')

    fig, axs = plt.subplots(figsize=(15, 3), ncols=2, nrows=1)
    axs[0].set_title(f'Визиты {dfs[2][1]} (воск.-суб.)')
    axs[0].plot(dfs[2][0].index, dfs[2][0]['client_id'], color='green')
    axs[1].set_title(f'Визиты {dfs[3][1]}')
    axs[1].plot(dfs[3][0].index, dfs[3][0]['client_id'], color='green')

    plt.show();


### Формирование полотна структуры посещений сайта
def structure_plot(dfs):
    """
    Функция визуализирует распределение визитов клиентов по географии, девайсам, каналам привлечения

        Параметры:
            dfs (DataFrame): cагрегированная часть датасета визитов, касающаяся распределения визитов по по географии, дивайсам, каналам привлечения
        Выходные параметры (None)

    """

    print('\nРАСПРЕДЕЛЕНИЕ КЛИЕНТОВ ПО ГЕОГРАФИИ, ДЕВАЙСАМ И КАНАЛАМ ПРИВЛЕЧЕНИЯ')
    for i in range(len(dfs)):
        fig, ax = plt.subplots(figsize=(15, 3))

        a = dfs[i][0][0]
        b = dfs[i][0][1]
        y_pos = np.arange(len(a))

        ax.barh(y_pos, b, color='green')
        ax.set_yticks(y_pos, labels=a)
        ax.invert_yaxis()
        ax.set_title(dfs[i][1])
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

        plt.show()