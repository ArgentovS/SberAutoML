import os
import pandas as pd
import datetime as dt
import missingno as msno
import matplotlib.pylab as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from scipy import stats


def pvalue_info(pvalue):
    """
        Функция вывода информации о параметре pvalue, полученного в результате статистического теста

            Параметры:
                pvalue (float): значение pvalue
            Выходные параметры (None)

    """

    print(f'pvalue = {(pvalue * 100):.6f}%')
    if pvalue > 0.05:
        print('Выбираем гипотезу H0')
    else:
        print('Выбираем гипотезу H1')


def paired_T_test(df_1, df_2, check):
    """
        Функция запуска теста по парному Т-критерию

            Параметры:
                df_1 (DataFrame): первая сравниваемая выборка
                df_2 (DataFrame): вторая сравниваемая выборка
            Выходные параметры (float)

    """
    if pd.isna(check):
        pvalue = stats.ttest_rel(df_1, df_2)[1]
        print('Выбранный тест: парный Т-критерий\n')
        pvalue_info(pvalue)
    else:
        pvalue = stats.ttest_rel(df_1, df_2, alternative='less')[1]
    return pvalue


def student_T_test(df_1, df_2, check):
    """
        Функция запуска теста по Т-критерию Стьюдента

            Параметры:
                df_1 (DataFrame): первая сравниваемая выборка
                df_2 (DataFrame): вторая сравниваемая выборка
            Выходные параметры (float)

    """
    if pd.isna(check):
        pvalue = stats.ttest_ind(df_1, df_2)[1]
        print('Выбранный тест: Т-критерий Стьюдента\n')
        pvalue_info(pvalue)
    else:
        pvalue = stats.ttest_ind(df_1, df_2, alternative='less')[1]
    return pvalue


def welch_T_test(df_1, df_2, check):
    """
        Функция запуска теста по Т-критерию Уэлча

            Параметры:
                df_1 (DataFrame): первая сравниваемая выборка
                df_2 (DataFrame): вторая сравниваемая выборка
            Выходные параметры (float)

    """
    if pd.isna(check):
        pvalue = stats.ttest_ind(df_1, df_2)[1]
        print('Выбранный тест: Т-критерий Уэлча\n')
        pvalue_info(pvalue)
    else:
        pvalue = stats.ttest_ind(df_1, df_2, alternative='less')[1]
    return pvalue


def wilcoxon_RANK_test(df_1, df_2, check):
    """
        Функция запуска теста по ранговому критерию Уилкоксона

            Параметры:
                df_1 (DataFrame): первая сравниваемая выборка
                df_2 (DataFrame): вторая сравниваемая выборка
            Выходные параметры (float)

    """
    if pd.isna(check):
        pvalue = stats.wilcoxon(df_1, df_2)[1]
        print('Выбранный тест: знаковый ранговый критерий Уилкоксона\n')
        pvalue_info(pvalue)
    else:
        pvalue = stats.wilcoxon(df_1, df_2, alternative='less')[1]
    return pvalue


def mannwhitney_U_test(df_1, df_2, check):
    """
        Функция запуска теста по критерию Манна-Уитни

            Параметры:
                df_1 (DataFrame): первая сравниваемая выборка
                df_2 (DataFrame): вторая сравниваемая выборка
            Выходные параметры (float)

    """
    if pd.isna(check):
        pvalue = stats.mannwhitneyu(df_1, df_2)[1]
        print('Выбранный тест: критерий Манна-Уитни\n')
        pvalue_info(pvalue)
    else:
        pvalue = stats.mannwhitneyu(df_1, df_2, alternative='less')[1]
    return pvalue


# Функция выбора статистического теста для проверки гипотезы
def shapiro_test(df_1, df_2, check):
    """
        Функция проверки двух выборок на распределение по нормальному закону

            Параметры:
                df_1 (DataFrame): первая проверяемая выборка
                df_2 (DataFrame): вторая проверяемая выборка
            Выходные параметры (bool)

    """
    s_df_1 = round(stats.shapiro(df_1)[1] * 100, 6)
    s_df_2 = round(stats.shapiro(df_2)[1] * 100, 6)
    if pd.isna(check):
        print(' ЗНАЧЕНИЯ pvalue ВЫБОРОК')
        print('  на тесте Шапира-Уилка')
        print('============================')
        print(f' Первая выборка: {s_df_1:.6f}%')
        print(f' Вторая выборка: {s_df_2:.6f}%')
        print('----------------------------')
    if s_df_1 <= .05 or s_df_2 <= 0.05:
        if pd.isna(check):
            print('Выборки НЕПАРАМЕТРИЧЕСКИЕ (не распеределены по нормальному закону распределения)')
        return False
    else:
        if pd.isna(check):
            print('Выборки ПАРАМЕТРИЧЕСКИЕ (распеределены по нормальному закону распределения)')
        return True


def choose_method(df_1, df_2, dependenceFlag=True, check=None):
    """
        Функция выбора теста для проверки гипотез

            Параметры:
                df_1 (DataFrame): первая сравниваемая выборка
                df_2 (DataFrame): вторая сравниваемая выборка
                dependenceFlag (bool): флаг зависимости двух выборок
                check (str): параметр указывающий необходимость проверки снижения статистической оценки второй выборки в сравнении с первой выборкой
            Выходные параметры (float)

    """
    # Проверяем распределения выборок по нормальному закону
    normalFlag = shapiro_test(df_1, df_2, check)
    # Указываем на зависимость или независимость выборок
    if pd.isna(check):
        if dependenceFlag:
            print('Выборки ЗАВИСИМЫЕ (парам в 1 выборке соответствуют пары во 2 выборке)\n')
        else:
            print('Выборки НЕ ЗАВИСИМЫЕ (парам в 1 выборке не соответствуют пары во 2 выборке)\n')

    # Выбор теста для проверки гипотезы
    if normalFlag and dependenceFlag:
        return paired_T_test(df_1, df_2, check)
    elif normalFlag and not dependenceFlag:
        if stats.levene(df_1, df_2)[1] <= 0.05:
            if pd.isna(check):
                print('ДИСПЕРСИИ выборок - неравны')
            return welch_T_test(df_1, df_2, check)
        else:
            if pd.isna(check):
                print('ДИСПЕРСИИ выборок - равны')
            return student_T_test(df_1, df_2, check)
    elif not normalFlag and dependenceFlag:
        return wilcoxon_RANK_test(df_1, df_2, check)
    else:
        return mannwhitney_U_test(df_1, df_2, check)















#
# def pvalue_info(pvalue):
#     """
#         Функция вывода информации о параметре pvalue, полученного в результате статистического теста
#
#             Параметры:
#                 pvalue (float): значение pvalue
#             Выходные параметры (None)
#
#     """
#
#     print(f'pvalue = {(pvalue * 100):.6f}%')
#     if pvalue > 0.05:
#         print('Выбираем гипотезу H0')
#     else:
#         print('Выбираем гипотезу H1')
#
#
# def paired_T_test(df_1, df_2):
#     """
#         Функция запуска теста по парному Т-критерию
#
#             Параметры:
#                 df_1 (DataFrame): первая сравниваемая выборка
#                 df_2 (DataFrame): вторая сравниваемая выборка
#             Выходные параметры (None)
#
#     """
#     print('Выбранный тест: парный Т-критерий\n')
#     pvalue = stats.ttest_rel(df_1, df_2)
#     pvalue_info(pvalue)
#
#
# def student_T_test(df_1, df_2):
#     """
#         Функция запуска теста по Т-критерию Стьюдента
#
#             Параметры:
#                 df_1 (DataFrame): первая сравниваемая выборка
#                 df_2 (DataFrame): вторая сравниваемая выборка
#             Выходные параметры (None)
#
#     """
#     print('Выбранный тест: Т-критерий Стьюдента\n')
#     pvalue = stats.ttest_ind(df_1, df_2)[1]
#     pvalue_info(pvalue)
#
#
# def welch_T_test(df_1, df_2):
#     """
#         Функция запуска теста по Т-критерию Уэлча
#
#             Параметры:
#                 df_1 (DataFrame): первая сравниваемая выборка
#                 df_2 (DataFrame): вторая сравниваемая выборка
#             Выходные параметры (None)
#
#     """
#     print('Выбранный тест: Т-критерий Уэлча\n')
#     pvalue = stats.ttest_ind(df_1, df_2, equal_var=False)
#     pvalue_info(pvalue)
#
#
# def wilcoxon_RANK_test():
#     """
#         Функция запуска теста по ранговому критерию Уилкоксона
#
#             Параметры:
#                 df_1 (DataFrame): первая сравниваемая выборка
#                 df_2 (DataFrame): вторая сравниваемая выборка
#             Выходные параметры (None)
#
#     """
#     print('Выбранный тест: знаковый ранговый критерий Уилкоксона\n')
#     pvalue = stats.wilcoxon(df_1, df_2)
#     pvalue_info(pvalue)
#
#
# def mannwhitney_U_test(df_1, df_2):
#     """
#         Функция запуска теста по критерию Манна-Уитни
#
#             Параметры:
#                 df_1 (DataFrame): первая сравниваемая выборка
#                 df_2 (DataFrame): вторая сравниваемая выборка
#             Выходные параметры (None)
#
#     """
#     print('Выбранный тест: критерий Манна-Уитни\n')
#     pvalue = stats.mannwhitneyu(df_1, df_2)[1]
#     pvalue_info(pvalue)
#
#
# # Функция выбора статистического теста для проверки гипотезы
# def shapiro_test(df_1, df_2):
#     """
#         Функция проверки двух выборок на распределение по нормальному закону
#
#             Параметры:
#                 df_1 (DataFrame): первая проверяемая выборка
#                 df_2 (DataFrame): вторая проверяемая выборка
#             Выходные параметры (bool)
#
#     """
#     s_df_1 = round(stats.shapiro(df_1)[1] * 100, 6)
#     s_df_2 = round(stats.shapiro(df_2)[1] * 100, 6)
#     print(' ЗНАЧЕНИЯ pvalue ВЫБОРОК')
#     print('  на тесте Шапира-Уилка')
#     print('============================')
#     print(f' Первая выборка: {s_df_1:.6f}%')
#     print(f' Вторая выборка: {s_df_2:.6f}%')
#     print('----------------------------')
#     if s_df_1 <= .05 or s_df_2 <= 0.05:
#         print('Выборки НЕПАРАМЕТРИЧЕСКИЕ (не распеределены по нормальному закону распределения)')
#         return False
#     else:
#         print('Выборки ПАРАМЕТРИЧЕСКИЕ (распеределены по нормальному закону распределения)')
#         return True
#
#
# def choose_method(df_1, df_2, dependenceFlag=True):
#     """
#         Функция выбора теста для проверки гипотез
#
#             Параметры:
#                 df_1 (DataFrame): первая сравниваемая выборка
#                 df_2 (DataFrame): вторая сравниваемая выборка
#                 dependenceFlag (bool): флаг зависимости двух выборок
#             Выходные параметры (None)
#
#     """
#     # Проверяем распределения выборок по нормальному закону
#     normalFlag = shapiro_test(df_1, df_2)
#     # Указываем на зависимость илим независимость выборок
#     if dependenceFlag:
#         print('Выборки ЗАВИСИМЫЕ (парам в 1 выборке соответствуют пары во 2 выборке)\n')
#     else:
#         print('Выборки НЕ ЗАВИСИМЫЕ (парам в 1 выборке не соответствуют пары во 2 выборке)\n')
#
#     # Выбор теста для проверки гипотезы
#     if normalFlag and dependenceFlag:
#         paired_T_test(df_1, df_2)
#     elif normalFlag and not dependenceFlag:
#         if stats.levene(df_1, df_2)[1] <= 0.05:
#             print('ДИСПЕРСИИ выборок - неравны')
#             welch_T_test(df_1, df_2)
#         else:
#             print('ДИСПЕРСИИ выборок - равны')
#             student_T_test(df_1, df_2)
#     elif not normalFlag and dependenceFlag:
#         wilcoxon_RANK_test(df_1, df_2)
#     else:
#         mannwhitney_U_test(df_1, df_2)
