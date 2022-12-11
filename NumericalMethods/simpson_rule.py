import numpy as np
from NumericalMethods.mesh_processing import split_into_areas, combine_areas
from GammaRayInteractions.Materials import ENERGY_MESH
from MeasurementInstrumentation.measurement_techniques import edges_indices

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


def simpson_rule(function, mesh=ENERGY_MESH, function_jumps=[], regime='s'):
    """
    Возвращает коэффициенты интегрирования по правилу Симпсона
    Параметры:
        1) function подынтегральная функция
        2) mesh сетка узлов
    """
    function_values_areas = split_into_areas(function) # подынтегральная функция разбитая на тройки
    energy_mesh_areas = split_into_areas(mesh) # сетка интегрирования разбитая на тройки
    # коэффициенты интерполяционного многочлена при интегрировании по правилу Симпсона
    integration_coefficients = polinom_coefficients_vectorize(energy_mesh_areas)
    # коэффициенты аппроксимации при интегрировании по правилу Симпсона
    approximation_coefficients = integration_coefficients*function_values_areas

    edges = edges_indices() # все индексы скачков
    for ind in function_jumps:
        edges.remove(ind) # удаление из списка скачков, которые нужно добавить при востановлении

    if regime == 's':
        return approximation_coefficients.sum()
    if regime == 'c':
        approximation_coefficients = combine_areas(approximation_coefficients, edges, mesh)[1]
        return approximation_coefficients
    if regime == 'm':
        approximation_mesh = combine_areas(approximation_coefficients, edges, mesh)[0]
        return approximation_mesh
