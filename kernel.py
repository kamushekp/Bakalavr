def kernel(u,mode='gauss',gama=0):
    if mode=='epanech':
        return{abs(u)>1+1e-14:0,abs(u)<=1+1e-14:0.75*(1-u*u)}[True]
    elif mode=='gauss':
        from math import pi,exp
        return exp(-u*u/2)*(0.5/pi)**0.5
    elif mode=='beta':
            from scipy import special
            return {abs(u)>1:0,
                    abs(u)<=1:(1-u*u)**gama/special.beta\
                    (0.5,gama+1)}[True]
    elif mode=='unif':
        return {abs(u)>1:0,abs(u)<=1:0.5}[True]


