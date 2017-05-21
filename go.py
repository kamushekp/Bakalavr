
import numpy as np
from find_design import find_best_design as fbd
kern = 'unif'


from M import M_symm
def main():
    print('kernel = {0}, p = {1}, h = {2}\n'.format(kern,p,h))
    resNM,resDIF=fbd(purpose,x0,p,h,kern)
    #resNM=fbd(purpose,x0,p,h,kern)
    print('\n\nNelder-Mead')
    
    res=0
    for i in range(len(purpose)):
        a=M_symm(purpose[i], resNM.x, kern, p, h)
        print(np.linalg.det(a))
        res+=np.linalg.det(a)
    
    print(res)
    print(list(sorted(resNM.x)))
    
    print('\n\nDiff. evolution')
    res=0
    for i in range(len(purpose)):
        a=M_symm(purpose[i], resDIF.x, kern, p, h)
        print(np.linalg.det(a))
        res+=np.linalg.det(a)
    
    print(res)
    print(list(sorted(resDIF.x)))
    

def get_graphics_of_parts():
    #Рисует график вкладов в определитель
    X_ = []
    Y_epan = []
    Y_gauss =  []  
    
    for step in np.linspace(0.4, 7, 100):
        kern='gauss'
        print(step)
        resNM, resDIF=fbd(purpose,x0,p,step,kern)
        dets = [np.linalg.det(M_symm(purpose[i], resNM.x, kern, p, step)) for i in range(3)]
        Y_epan.append(max(dets) / sum(dets))
        
    for step in np.linspace(0.4, 7, 100):
        kern='epanech'
        print(step)
        resNM, resDIF=fbd(purpose,x0,p,step,kern)
        dets = [np.linalg.det(M_symm(purpose[i], resDIF.x, kern, p, step)) for i in range(3)]
        Y_gauss.append(max(dets) / sum(dets))
        X_.append(step)
    print('\n\nGauss - {0}\n\nEpanech - {1}'.format(Y_gauss, Y_epan))


    
    f = open('parts.txt', 'w')
    for elem in zip(X_, Y_epan, Y_gauss):
        f.write(str(elem[0])+','+str(elem[1])+','+str(elem[2])+'\n')
'''
get_graphics_of_parts()
gauss = [1.0, 1.0, 0.99999999999744971, 0.99999999999305167, 0.99999999696605468, 0.99999999997653211, 0.9999999999687913, 0.69870841313358345, 0.64397724308302273, 0.60926518722983081, 0.58736190794084064, 0.50000003972560869, 0.50000386595569635, 0.50000872685271047, 0.50002052372238448, 0.53817189990789882, 0.53611491299350211, 0.51568367497945633, 0.49828627712306689, 0.48335340459507481, 0.47041165209585822, 0.45911323671512261, 0.44918158602779829, 0.44040753954896866, 0.43260513099776138, 0.42564436020452423, 0.41939915755785201, 0.41377357977365015, 0.40868366453821392, 0.40407384711639499, 0.39987363834764256, 0.39603912236539457, 0.39253017174802912, 0.38930805484728881, 0.38634405902799251, 0.38360943626068306, 0.38107886320390827, 0.37874192606279894, 0.37656968027646759, 0.37455385051664986, 0.37266923566826993, 0.37091387504017731, 0.36927368543478839, 0.36773975060441849, 0.36629695370742305, 0.3649491662661683, 0.36368125574851073, 0.36248589039722973, 0.36135957118652096, 0.36030100864697945]
epanech = [0.52937303553103465, 0.54230253200222012, 0.56168259257458353, 0.52177379077123986, 0.47985664045001425, 0.54355306056526076, 0.49798615676515434, 0.4654059568967262, 0.44149548150761919, 0.4234972248237649, 0.40963579508726172, 0.39874320627140925, 0.39003160440910406, 0.38295623762412273, 0.37713163096126584, 0.37227919291879619, 0.36819369388394863, 0.36472131860871809, 0.36174497870279898, 0.35917428126195139, 0.35693854745358861, 0.35498186893127676, 0.3532595511458963, 0.35173551679481058, 0.35038038454675074, 0.34917002968460453, 0.34808449335797631, 0.3471071471957769, 0.34622404716405447, 0.34542342920011682, 0.34469531213920335, 0.34403118260699239, 0.34342374308488588, 0.34286670906844313, 0.34235464467495563, 0.34188282858484004, 0.34144714407980709, 0.34104398834802868, 0.34067019728951409, 0.34032298286408197, 0.33999988064476577, 0.33969870571856958, 0.33941751544883847, 0.33915457790470366, 0.33890834499218636, 0.33867742950281465, 0.33846058543982499, 0.33825669109733614, 0.33806473446055724, 0.33788380056990835]
x_axe=np.linspace(0.4, 7, 50)
import matplotlib.pyplot as plt
plt.plot(x_axe, epanech,'.', x_axe, gauss,'g^')
plt.show()
'''
main()

 
