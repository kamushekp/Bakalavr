import numpy as np
from kernel import kernel as K


def get_f(purpose,x,p):
    return np.matrix([(x-purpose)**(j) for j in range(p+1)]).T


def M_symm(purpose, design, kernel, p, h, odd):

    n=len(design)
    Fish = np.zeros((p + 1, p + 1))
    Kui = [K((design[i] - purpose) / h, mode=kernel) / h for i in range(n)]
    if odd == 0:#число точек - четное
        for i in range(p + 1):
            for j in range(p + 1):
                s = i + j
                if s % 2 == 1:
                    Fish[i, j] += 0
                else:
                    Fish[i, j] += 2 * sum([(design[a] - purpose) ** s * Kui[a] for a in range(n)])

    else:#число точек - нечетное, и последняя в плане - не имеет парную
        for i in range(p+1):
            for j in range(p+1):
                s = i+j
                if s % 2 == 0:
                    Fish[i, j] += 2 * sum([(design[a] - purpose)**s*Kui[a] for a in range(n - odd)])
                Fish[i, j] += Kui[n - 1] * (design[n - 1] - purpose) ** s

    return Fish

def M(purpose,design,kernel,p,h):
    n=len(design)
    M=np.zeros((p+1,p+1))
    Kui=[K((design[i] - purpose)/h, mode=kernel) / h for i in range(n)]
    for i in range(n):
        f_x=get_f(purpose,design[i],p)
        M+=Kui[i]*np.dot(f_x,f_x.T)
    return M

if __name__ == '__main__':
    resM = 0
    resMsymm = 0
    purps = [2]
    for e in purps:

        a = M(e,[1,1.5,2.5,3,2],'gauss', 2, 2)
        resM += np.linalg.det(a)
        print(a)
        print()

        b = M_symm(e, [1, 1.5,2], 'gauss', 2, 2, odd=1)
        resMsymm += np.linalg.det(b)
        print(b)
    print(resM, resMsymm)
