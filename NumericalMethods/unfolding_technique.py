import numpy as np
from NumericalMethods.simpson_rule import simpson_rule

def get_deviation(initial_mesh, initial_spectrum, deconvolved_mesh, deconvolved_spectrum):
    """
    Возвращает ошибку восстановления спектра
    """
    deconvolved_spectrum_on_initial_mesh = np.interp(initial_mesh, deconvolved_mesh, deconvolved_spectrum)
    unfolding_deviation = (initial_spectrum - deconvolved_spectrum_on_initial_mesh)**2

    return simpson_rule(unfolding_deviation) / simpson_rule(initial_spectrum**2)


def get_deviation_up_to_bound(initial_mesh, initial_spectrum, deconvolved_mesh, deconvolved_spectrum, bound):
    """
    Возвращает ошибки восстановления спектра в области до К скачка 100 элемента таблицы Менделеева
    """
    deconvolved_spectrum_on_initial_mesh = np.interp(initial_mesh, deconvolved_mesh, deconvolved_spectrum)
    initial_spectrum_up_to_bound = np.copy(initial_spectrum)
    deconvolved_spectrum_up_to_bound = np.copy(deconvolved_spectrum_on_initial_mesh)
    initial_spectrum_up_to_bound[bound:] = 0 # зануление части, лежащей выше требуемого К скачка
    deconvolved_spectrum_up_to_bound[bound:] = 0 # зануление части, лежащей выше требуемого К скачка
    unfolding_deviation = (initial_spectrum_up_to_bound - deconvolved_spectrum_up_to_bound)**2

    return simpson_rule(unfolding_deviation) / simpson_rule(initial_spectrum_up_to_bound**2)


def Gold(matrix, results, weight, initial_mesh, deconvolved_mesh, initial_spectrum, iteration_number, bound, journal=False, weight_mode='value'):
    A, b = matrix, results
    if weight_mode == "value":
        W = np.diag([float(val**weight) for i,val in enumerate(b)])
    if weight_mode == "channel":
        W = np.diag([float(i**weight) for i,val in enumerate(b)])
    previous_x = np.ones((A.shape[1])) # начальное приближение
    current_x = np.ones((A.shape[1])) # текущее значение

    signals_deviation = np.empty(iteration_number) # пустой массив под запись отклонения сигналов на каждой итерации
    unfolding_deviation = np.empty(iteration_number) # пустой массив под запись ошибки восстановления спектра
    unfolding_deviation_up_to_100_kev = np.empty(iteration_number) # пустой массив под запись ошибки восстановления спектра до 100 кэв

    bound_100_kev = bound

    if journal == False:
        # Основной цикл алгоритма Голда
        for k in range(0, iteration_number + 1):
            # Создание матрицы Y = A.T * W.T * W * b
            Y = np.dot(A.T, np.dot(W.T, np.dot(W, b)))
            # Создание матрицы AX = A.T * W.T * W * A * x
            AX = np.dot(A.T, np.dot(W.T, np.dot(W, np.dot(A, previous_x))))
            AX[AX == 0] = np.nextafter(0, 1)*1e20
            current_x = previous_x + (previous_x / AX) * (Y - AX)
            previous_x = current_x

        unfolding_deviation = get_deviation(initial_mesh, initial_spectrum, deconvolved_mesh, current_x)
        unfolding_deviation_up_to_100_kev = get_deviation_up_to_bound(initial_mesh, initial_spectrum, deconvolved_mesh, current_x, bound_100_kev)
        signals_deviation = np.linalg.norm(np.dot(A, current_x) - b)

        return current_x, unfolding_deviation, unfolding_deviation_up_to_100_kev, signals_deviation


    if journal == True:

        # Основной цикл алгоритма Голда
        for k in range(0, iteration_number):
            # Создание матрицы Y = A.T * W.T * W * b
            Y = np.dot(A.T, np.dot(W.T, np.dot(W, b)))
            # Создание матрицы AX = A.T * W.T * W * A * x
            AX = np.dot(A.T, np.dot(W.T, np.dot(W, np.dot(A, previous_x))))
            AX[AX == 0] = np.nextafter(0, 1)*1e20
            current_x = previous_x + (previous_x / AX) * (Y - AX)
            previous_x = current_x

            signals_deviation[k] = np.linalg.norm(np.dot(A, current_x) - b)
            unfolding_deviation[k] = get_deviation(initial_mesh, initial_spectrum, deconvolved_mesh, current_x)

    return current_x, [list(range(iteration_number)), signals_deviation, unfolding_deviation]
