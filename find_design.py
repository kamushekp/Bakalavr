from scipy.optimize import differential_evolution, minimize
from M import *

def find_best_design(mode, method, len_purpose, x0_count, p, h, kernel, file_name, *args):
    """ Умножает длину начального приближения на количество целевых точек!!!
    :param mode: стандартное вычисление информационных матриц, либо с использованием модификаций
    :param method: Нелдер-Мид или дифференциальная эволюция
    :param len_purpose: количество целевых точек
    :param x0_count: количество точек вокруг каждой целевой
    :param p: последняя степень члена ряда Тейлора (всего P+1)
    :param h: ширина окна сглаживания
    :param kernel: ядерная функция
    :param file_name: файл для сохранения результатов
    :param args: целевые точки и начальное приближение
    :return: 
    """
    """
     count_finite_det(design, one_purpose) - считает определитель информационной матрицы
      для плана 'design' вокруг точки 'one_purpose'
     count_general_sum(design) - считает сумму определителей для всех целевых точек
     
     count_symm_finite_det(design, one_purpose) - считает определитель информационной матрицы
      для плана 'design' вокруг точки 'one_purpose' с использованием модификации
     count_symm_general_sum(design) - считает сумму определителей для всех целевых точек,
      посчитанных с модификациями
    """


    def count_symm_finite_det(design, one_purpose):
        return np.linalg.det(M_symm(purpose=one_purpose, design=design,
                                    kernel=kernel, p=p, h=h, odd=odd))

    def count_symm_general_sum(design):
        return -sum([count_finite_det(design, purp) for purp in purpose])

    def count_finite_det(design, one_purpose):
        return np.linalg.det(M(purpose=one_purpose, design=design,
                               kernel=kernel, p=p, h=h))

    def count_general_sum(design):
        return -sum([count_finite_det(design, purp) for purp in purpose])

    

    purpose = [float(e) for e in list(args[:int(len_purpose)])]
    p=int(p)
    h=float(h)
    x0=[]
    for purp in purpose:
        x0.extend(list(np.linspace(purp-h, purp+h, x0_count)))
    n = len(x0)

    if mode == 'modified':
        lastDotInDesign = x0[n - 1]
        x0 = x0[:n // 2]
        odd = 0
        if n % 2 == 1:
            x0.append(lastDotInDesign)
            odd = 1

        if method == 'DiffEvo':
            bounds = [(min(purpose) - 4 * h, max(purpose) + 4 * h) for i in range(len(x0))]
            res = differential_evolution(count_symm_general_sum, bounds)
        elif method == 'NelMead':
            res = minimize(count_symm_general_sum,x0,method='Nelder-Mead',\
        tol=1e-3,options={'maxiter': 1e+8, 'maxfev': 1e+8})
        def transform(design, purp):
            '''
            :param "симметричный" план эксперимента, построенный с использованием модификации 
            :return: план эксперимента в привычном понимании. Образован путем отражения точек
             из симметричного плана вокруг каждой из целевых
            '''
            if odd:
                l = [[purp - abs(e-purp), purp + abs(e-purp)] for e in design[:len(design) - 1]]
                return [elem for sublist in l for elem in sublist] + [design[len(design) - 1]]
            else:
                l = [[purp - abs(e-purp), purp + abs(e-purp)] for e in design]
                return [elem for sublist in l for elem in sublist]

        result = np.array([elem for sublist in [transform(res.x, purp) for purp in purpose] for elem in sublist])

    elif mode == 'standart':
        if method == 'DiffEvo':
            bounds = [(min(purpose) - 4 * h, max(purpose) + 4 * h) for i in range(len(x0))]
            res = differential_evolution(count_general_sum, bounds)
        elif method == 'NelMead':
            res = minimize(count_general_sum,x0,method='Nelder-Mead',\
        tol=1e-3,options={'maxiter': 1e+8, 'maxfev': 1e+8})

        result = np.array(res.x)
    np.savetxt(file_name, result, delimiter=' ', newline=' ')
    return result

if __name__=='__main__':
    from sys import argv as a
    find_best_design(a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],*a[9:len(a)])
    #find_best_design(mode , method, len_purpose, len_x0, p, h, kernel, file_name, purpose,x0)
    #find_best_design('standart', 'DE' ,'3', '3','2', '1' ,'gauss','C:\\Users\\Павел\\Desktop\\Article Kamenev\\test_planning_results.txt','0', '1', '2' )

'''time research
    n = 21

    p=2
    h=2
    kernel='gauss'
    file_name='res.txt'
    purpose=[1]
    len_purpose=len(purpose)
    method = 'DE'
    from time import time

    time_modef = []
    time_std = []
    for x0s in range(3, 20):
        print(x0s)
        x0 = list(np.linspace(0, 2, x0s))
        len_x0 = len(x0)
        start = time()
        for i in range(n-x0s):
            a=find_best_design('modef', method, len_purpose, len_x0, p, h, kernel, file_name, *purpose, *x0)
        end = time()
        time_modef.append((end-start)/n)

        start=time()
        for i in range(n-x0s):
            find_best_design('standart', method, len_purpose, len_x0, p, h, kernel, file_name, *purpose, *x0)
        end = time()

        time_std.append((end - start) / n)

        import matplotlib.pyplot as plt
        x_axe = list(i+3 for i,e in enumerate([1]*len(time_std)))
    plt.plot(x_axe, time_std, '--', x_axe, time_modef, '-')
    plt.show()

    np.savetxt(file_name, (time_std, time_modef), delimiter=' ', newline=' ')
'''





