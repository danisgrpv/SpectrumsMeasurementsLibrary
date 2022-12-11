import numpy as np

def split_into_areas(array):
    """
    Разделяет массив на части по три элемента
    (нужно переделать функцию)
    """
    triplets_num = len(array) / 3
    return np.array(np.split(array, triplets_num))


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

    return np.array(result_m), np.array(result_f)
