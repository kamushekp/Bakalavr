from scipy.optimize import minimize
import numpy as np
from M import M_symm, M
def find_best_design(mode, method, purpose, x0, p, h, kernel, file_name):
    n = len(x0)
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

    from scipy.optimize import differential_evolution, minimize
    bounds = [(min(purpose) - 4 * h, max(purpose) + 4 * h) for i in range(len(x0))]

    if mode != 'standart':
        last = x0[n - 1]
        x0 = x0[:n // 2]
        odd = 0
        if n % 2 == 1:
            x0.append(last)
            odd = 1

        '''res=minimize(count_finite_det,x0,method='Nelder-Mead',\
        tol=1e-3,options={'maxiter': 1e+8, 'maxfev': 1e+8})'''

        if method != 'NM':
            res = differential_evolution(count_symm_general_sum, bounds)
        else:
            res = minimize(count_symm_general_sum,x0,method='Nelder-Mead',\
        tol=1e-3,options={'maxiter': 1e+8, 'maxfev': 1e+8})
        def transform(design, purpose):
            '''
            :param design: 
            :return: transformed symmetrical around purpose point into standart design
            '''
            if odd:
                l = [[purpose - abs(e-purpose), purpose + abs(e-purpose)] for e in design[:len(design) - 1]]
                return [elem for sublist in l for elem in sublist] + [design[len(design) - 1]]
            else:
                l = [[purpose - abs(e-purpose), purpose + abs(e-purpose)] for e in design]
                return [elem for sublist in l for elem in sublist]
        result = np.array([elem for sublist in [transform(res.x, purp) for purp in purpose] for elem in sublist])

    else:
        if method != 'NM':
            res = differential_evolution(count_general_sum, bounds)
        else:
            res = minimize(count_general_sum,x0,method='Nelder-Mead',\
        tol=1e-3,options={'maxiter': 1e+8, 'maxfev': 1e+8})

        result = np.array(res.x)

    np.savetxt(file_name, result, delimiter=' ', newline=' ')
if __name__=='__main__':
    from sys import argv as a
    find_best_design(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7])
    #find_best_design(mode = 'modef', method = 'NM', purpose=[1], x0=[-1, 1, -0.5], p=1, h=2, kernel='unif', file_name='res.txt')





