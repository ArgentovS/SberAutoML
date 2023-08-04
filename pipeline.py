import sqlite3 as bd

import dill
import pandas as pd
import datetime as dt

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, FunctionTransformer


# Функция формирования фичей для обучения модели
def generate_basic_features(df):
    # create is_* features if missing
    if 'is_organic' not in df.columns:
        df['is_organic'] = [*map(lambda x: True if x in ['organic', 'referral', '(none)'] else False,
                                 df['utm_medium'].values)]
    else:
        pass

    if 'is_mobile' not in df.columns:
        df['is_mobile'] = [*map(lambda x: True if x in ['mobile'] else False,
                                df['device_category'].values)]
    else:
        pass

    if 'is_represented' not in df.columns:
        df['is_represented'] = [*map(lambda x: True if x in ['Moscow', 'Saint Petersburg', 'Balashikha',
                                                             'Khimki', 'Odintsovo', 'Vidnoye', 'Mytishchi',
                                                             'Zheleznodorozhny', 'Domodedovo', 'Korolyov'] else False,
                                     df['geo_city'].values)]
    else:
        pass

    if 'is_social' not in df.columns:
        df['is_social'] = [*map(lambda x: True if x in ['QxAxdyPLuQMEcrdZWdWb', 'MvfHsxITijuriZxsqZqt',
                                                        'ISrKoXQCxqqYvAZICvjs', 'IZEXUFLARCUMynmHNBGo',
                                                        'PlbkrSYoHuZBWfYjYnfw', 'gVRrcxiDQubJiljoTbGm'] else False,
                                df['utm_source'].values)]
    else:
        pass

    # replace is_* with 0\1
    for col in ['is_organic', 'is_mobile', 'is_represented', 'is_social']:
        df[col] = df[col].astype(int)

    # time features
    df['visit_year'] = df['visit_date'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d').year)
    df['visit_month'] = df['visit_date'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d').month)
    df['visit_day'] = df['visit_date'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d').day)
    df['visit_weekday'] = df['visit_date'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d').weekday())
    df['visit_hour'] = df['visit_time'].apply(lambda x: dt.datetime.strptime(x, '%H:%M:%S.%f').hour)
    print(df.info())

    return df


# Прочитаем из хранилища датасет sessions
connection = bd.connect('session.db')
df = pd.read_sql("SELECT * FROM table_sessions", connection)
connection.close()

# Сбалансируем датасет уменьшением количества негативного класса
# Выбирем все записи положительного класса в таргете
df_1 = df[df.conversion_rate == 1]
# Выберем записи отрицательного класса количесвтом, равным количеству записей положительного класса
df_0 = df[df.conversion_rate == 0].sample(int(len(df_1)))
# Объединим полученные записи положительного и отрицательного класса в сблансированный датасет
df_balance = pd.concat([df_1, df_0], axis=0, ignore_index='ignor')

# Приготовим данные для обучения
X = df_balance.drop(['session_id', 'visit_number', 'client', 'conversion_rate'], axis=1)
y = df_balance['conversion_rate']

# Подготовим json файлы для тестов работы модели через FastAPI
sample = X.sample(1)
for i in range(1 ,3):
    json_file = sample.to_json(orient='records')
    with open(f'data_{i}.json', 'w') as outfile:
        outfile.write(json_file)

# Объявим экземпляры классов для преобразования числовых и категориальных переменных, а также для обучения
scaler1 = StandardScaler()
scaler2 = StandardScaler()
oe = OrdinalEncoder()
rf = RandomForestClassifier(max_features='sqrt', min_samples_leaf=13, n_estimators=700, random_state=42)

# Сделаем pipeline для кодирования и стандартизации категориальных переменных датасета

def func1(df):
    cat_features = ['utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent', 'utm_keyword',
                'device_category', 'device_os', 'device_brand', 'device_screen_resolution', 'device_browser',
                'geo_country', 'geo_city']
    return df[cat_features].fillna('нет данных')

def func2(data):
    cat_features = ['utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent', 'utm_keyword',
                'device_category', 'device_os', 'device_brand', 'device_screen_resolution', 'device_browser',
                'geo_country', 'geo_city']
    return pd.DataFrame(data, columns=cat_features)

cat_features_selector = FunctionTransformer(func=func1, validate=False)
df_cat =  FunctionTransformer(func=func2)
cat_features_preprocessor = Pipeline([("cat_features_selector", cat_features_selector),
                                      ("oe", oe), ('scaler1', scaler1), ('df_cat', df_cat)])


# Сделаем pipeline для обогощения датасета фичами
new_features_selector = FunctionTransformer(func=generate_basic_features, validate=False)
new_features_preprocessor = Pipeline([("new_features_selector", new_features_selector)])


# Сделаем pipeline для стандартизации новых фичей
def func3(df):
    other_features = ['is_organic', 'is_mobile', 'is_represented', 'is_social',
                   'visit_year', 'visit_month', 'visit_day', 'visit_weekday', 'visit_hour']
    return df[other_features]


def func4(data):
    other_features = ['is_organic', 'is_mobile', 'is_represented', 'is_social',
                  'visit_year', 'visit_month', 'visit_day', 'visit_weekday', 'visit_hour']
    return pd.DataFrame(data, columns=other_features)


other_features_selector = FunctionTransformer(func=func3)
df_other =  FunctionTransformer(func=func4)
other_features_preprocessor = Pipeline([("new_features_preprocessor", new_features_preprocessor),
                                        ("other_features_selector", other_features_selector),
                                        ('scaler2', scaler2), ('df_other', df_other)])


# Установим диаграмное отображение объектов sklearn
sklearn.set_config(display='diagram')
# Объединим созданные выше pipeline в один с помощью функции FeatureUnion
# и затем записываем итоговый pipeline для модели "Случайный лес деревьев"
feature_union = FeatureUnion([("cat_features_preprocessor", cat_features_preprocessor),
("other_features_preprocessor", other_features_preprocessor)])
pipeline = Pipeline([("preprocessing", feature_union), ('rf', rf)])


# Из-за возможной, связанной при разряженности данных
# обучим модель на всех данных и проверим её качество
pipeline.fit(X, y)

# Проверим качество модели
score = round(roc_auc_score(y, pipeline.predict(X))*100, 2)
print(f'Метрика ROC AUC: {score}%')

# Упакуем модель в словарь
model = {
         'best_model': pipeline,
         'metadata':   {
                        'name': 'Модель предсказания целевых действий',
                        'author':  'Argentov Sergey',
                        'date':     dt.datetime.strftime(dt.datetime.now(), '%Y-%m-%d'),
                        'version': 'v.1.98',
                        'type': type(pipeline.named_steps['rf']).__name__,
                        'score': f'{score}%'
                       },
        }


# Записываем модель в формат pickle
with open('model.pickle', 'wb') as file:
    dill.dump(model, file, recurse=True)

# Считываем модель для проверки предсказания по json-файлу
with open('model.pickle', 'rb') as file:
    model = dill.load(file)
df_sample = pd.read_json('./data_1.json', orient='records')
# Исключим некорректное преобразование времени из json в датафрейм
df_sample['visit_time'] = df_sample['visit_time'].apply(lambda x: dt.datetime.strftime(x, '%H:%M:%S.%f'))

print(model['metadata'])
