## Черновики функций

### Агрегация параметров датасета визитов во времени
def visual_plots(df):
    # Подготовка полей для группировки датасета
    df['client'] = df['client_id']

    # Словарь агрегирующих функций по атрибутам и агрегация для визуализации
    ag_dict = {'client_id': 'max', 'visit_date': 'min', 'visit_time': 'min',
               'geo_country': 'max', 'geo_city': 'max',
               'device_category': 'max', 'device_browser': 'max',
               'utm_source': 'max', 'utm_medium': 'max'}

    df_ag = df.groupby('client').agg(ag_dict)

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

    return [(ag_ym, 'по мясецам'),
            (ag_dm, 'по дням месяца'),
            (ag_dw, 'по дням недели'),
            (ag_hd, 'по часам дня')]


### Визуализация параметров датасета визитов во времени
def show_plot(df, title):
    fig, bx = plt.subplots(figsize=(12, 3))
    bx.plot(df.index, df['client_id'], color='green')
    bx.set_xlabel('Периоды')
    bx.set_ylabel('Количество уникальных визитов')
    bx.set_title(f'Распределение визитов {title}')
    bx.yaxis.set_major_formatter(FormatStrFormatter('%.0f'));


def fig_hist(dfs):
    for elem in dfs:
        show_plot(elem[0], elem[1])


### Функция конкатенации строки даты и времени
def create_date_time_visit(date, time):
    """
    Функция сцепляет строковое значение даты и времени визитов.
    Предназнаена для использования с датасетом визитов

        Параметры:
            date (str): дата визита
            time (str): время визита
        Выходные параметры (str, None)

    """
    if pd.notna(date) and pd.notna(time):
        return str(date) + ' ' + str(time)
    elif pd.notna(date):
        return str(date) + str(' 00:00:00')
    else:
        return None


### Функция обогощения времени визита и времени каждого события (в наносекундах)
def create_date_time_ns(date_time, date, ns):
    """
    Функция обогащает время визита при его отсутсвии в датасете визитов и сцепляет строковое значение времени событий и визитов.
    Предназнаена для использования с объединённым датасетом

        Параметры:
            date_time (str): дата и время визита
            data (str): дата события
            ms (str): время (в наносекундах)
        Выходные параметры (str, None)

    """
    if pd.notna(date_time) and pd.notna(ns):
        return str(date_time) + f'{int(ns) / 1000000000:.9f}'[1:]
    elif pd.notna(date_time):
        return str(date_time) + '.000000000'
    elif pd.isna(date_time) and pd.notna(date) and pd.notna(ns):
        return str(date) + f' 00:00:0{int(ns) / 1000000000:.9f}'
    elif pd.isna(date_time) and pd.notna(date) and pd.isna(ns):
        return str(date) + ' 00:00:00.000000000'
    else:
        return None


### Функция объединения датасетов
def data_marge(df_pk, df_fk, key):
    """
    Функция объединяет два датасета и иллюстрирует пропуски данных

        Параметры:
            df_pk (DataFrame): датасет, с первичным ключом, используемым для объединени
            df_fk (DataFrame): датасет, с внешним ключом, используемым для объединения
        Выходные параметры:
            df (DataFrame): объединённый датасет

    """
    print(f"\n... объединяем датасеты по ключу '{key}' ...")
    df = df_pk.merge(df_fk, left_on=key, right_on=key, how='outer')
    print('Датасеты объединёны.')
    print('... очищаем память от загруженных ранее промежуточных датасетов ...')
    del df_pk, df_fk  # очищаем память от ненужных датасетов
    print('Промежуточные датасеты удалены.\n')

    print('\n... запускаем объединение полей даты и времени события ...')
    df['date_time'] = df.apply(lambda x:
                               create_date_time_ns(x.date_time, x.hit_date, x.hit_time),
                               axis=1)
    print("Объединение полей даты и времени событий осуществлено в поле 'date_time'.")

    print("\n... запускаем преобразование типа в полей 'date_time' ...")
    df['date_time'] = pd.to_datetime(df['date_time'])
    print("Преобразование типа в поле 'date_time' на тип `datetime64(ns)` завершено .")

    df.drop(columns=['hit_date', 'hit_time'], inplace=True)
    print("Поля 'hit_date' и 'hit_time' удалены из объединённого датасета.")
    size_df = df.shape
    print(f'Датасет содержит {size_df[0]} строк и {size_df[1]} столбцов.')

    print('\n... анализируем пропущенные значения в датасете ...')
    data_set_audit(df)  # запускаем проверку на пустые значения в датасете
    print('Анализ пропущенных значений в датасете завершён.\n')

    print('... готовим иллюстрацию заполненности датасета значениями ...')
    msno.matrix(df)  # визуализируем заполнение занчениями датасета

    return df