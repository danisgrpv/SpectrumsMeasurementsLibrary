import numpy as np

def split_into_areas(array):
    """
    Разделяет массив на части по три элемента
    (нужно переделать функцию)
    """
    triplets_num = len(array) / 3
    return np.array(np.split(array, triplets_num))


def polinom_coeff(mesh_triplet):
    """
    Принимает массив со значениями узлов элементарного участка интегрирования
    по правилу Симпсона [x0, x1, x2]
    Возвращает коэффициенты интегрирования по правилу Симпсона [c0, c1, c2]
    Для произвольной сетки узлов
    """
    x = mesh_triplet
    x21 = x[2]-x[1]
    x20 = x[2]-x[0]
    x10 = x[1]-x[0]
    c0 = (6*x20*x10 + 2*x20**2 - 3*x20*(x10+x20)) / (6*x10)
    c1 = x20**3 / (6*x10*x21)
    c2 = (x20*(2*x20 - 3*x10)) / (6*x21)
    return np.array([c0, c1, c2])


def polinom_coeff_h(mesh_triplet):
    """
    Принимает массив со значениями узлов элементарного участка интегрирования
    по правилу Симпсона [x0, x1, x2]
    Возвращает коэффициенты интерполяционного многочлена при
    интегрировании по правилу Симпсона [c0, c1, c2]
    Для сетки узлов с равномерным шагом
    """
    x = mesh_triplet
    x21 = x[2]-x[1]
    x20 = x[2]-x[0]
    x10 = x[1]-x[0]
    c0 = (1/6)*x20
    c1 = (4/6)*x20
    c2 = (1/6)*x20
    return np.array([c0, c1, c2])


def polinom_coefficients_vectorize(split_areas):
    """
    Принимает массив с тройками узлов [[x0, x1, x2], [x3, x4, x5],... [xn-2, xn-1, xn]]
    и возвращает подобный массив, но с коэффициентами интерполяционного многочлена при
    интегрировании по правилу Симпсона
    [[с0, с1, с2], [с3, с4, с5],... [сn-2, сn-1, сn]]
    Является векторизированным вариантом фунции polinom_coeff
    """
    return np.array(list(map(polinom_coeff, split_areas)))


def concatenate_function_list(a, b):
    """
    Соединяет две области, сшивая точку разрыва
    [1,2,3] [3,4,5] -> [1,2,6,4,5]
    """
    return a[:len(a) -1] + [a[-1] + b[0]] + b[1:]


def concatenate_mesh_list(a, b):
    """
    [1,2,3] [3,4,5] -> [1,2,3,3,4,5]
    """
    return a[:len(a) -1] + [b[0]] + b[1:]


def combine_areas(areas, jumps_list, mesh):
    """
    Возвращает энергетическую сетку для восстановления спектра и значения коэффициентов в ее узлах
    Параметры:
        1) areas тройки значений фунции
        2) jumps_list индексы скачков, которые нужно добавить при восстановлении
        3) энергетическая сетка
    """
    if isinstance(areas, np.ndarray):
        areas = areas.tolist()

    mesh_areas = split_into_areas(mesh)
    mesh_areas = mesh_areas.tolist()

    result_f = areas[0]
    result_m = mesh_areas[0]

    for i, part in enumerate(areas):
        if i < len(areas) - 1:
            if mesh.tolist().index(mesh_areas[i][-1]) in jumps_list:
                result_f = concatenate_function_list(result_f, areas[i + 1])
                result_m = concatenate_mesh_list(result_m, mesh_areas[i + 1])
            else:
                result_f += areas[i + 1]
                result_m += mesh_areas[i + 1]

    return result_m, result_f
