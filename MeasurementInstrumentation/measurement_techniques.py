import numpy as np
from SpectrumsMeasurementsLibrary.GammaRayInteractions.Materials import Material, R


def k_edge_method_transmission_function(material_number, transmission_in_k_edge):
    """
    Возвращает фунцию пропускания канала системы, основанной на методе краевых фильтров
    Параметры:
        1) Зарядовый номер материала фильтра
        2) Необходимый урововень пропускания в К крае поглощения
    """
    region_thiknesses = Material(material_number).thickness_for_level(transmission_in_k_edge)
    region = R(material_number, region_thiknesses)
    return region.transmission()


def ross_method_transmission_function(first_material_number, second_material_number, transmission_in_k_edge):
    """
    Возвращает фунцию пропускания канала системы, основанной на методе фильтров Росса
    Параметры:
        1) Зарядовый номер материала вычетного (номер один) и опорного (номер два) фильтра
        2) Необходимый урововень пропускания в К крае поглощения опорного фильтра
    """
    second_filter_transmission = k_edge_method_transmission_function(second_material_number, transmission_in_k_edge)
    second_filter_transmission_in_first_filter_k_edge = second_filter_transmission[Material(first_material_number).k_edge_index()]
    first_filter_transmission = k_edge_method_transmission_function(first_material_number, second_filter_transmission_in_first_filter_k_edge)
    
    return second_filter_transmission - first_filter_transmission


def ideal_transmission_function(first_material_number, second_material_number, transmission_level):
    """
    Возвращает фунцию пропускания канала, являющуюся идеальной полосой пропускания с границами
    между К краями указанных материалов
    Параметры:
        1) Зарядовый номер материала вычетного (номер один) и опорного (номер два) фильтра
        2) Необходимый урововень пропускания в К крае поглощения опорного фильтра
    """
    energy_mesh = Material(first_material_number).mesh()
    transmission = np.zeros(energy_mesh.shape[0])
    passband_left_edge = Material(first_material_number).k_edge_index()
    passband_right_edge = Material(second_material_number).k_edge_index()
    transmission[passband_left_edge+1 : passband_right_edge+1] = transmission_level
    
    return transmission