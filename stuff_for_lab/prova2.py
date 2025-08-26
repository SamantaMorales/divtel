import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from scipy.optimize import curve_fit

def func(x, a, b):
    return b +a*x


plt.figure()

t2 = Table.read('samanta4500.out',format='ascii')

x=t2['EP'] #energ. minima 
y=t2['C']  #abbondanze calcolata
fe=t2['elem'] 
#z=t2['D'] #EW


z=100

#limit=(fe==26.0) #elimino righe troppo sature e molto piccole e scelgo le righe del ferro I
limit=(fe==26.0)
x=x[limit]
y=y[limit]


limit=((y<np.mean(y)+2*np.std(y))&(y>np.mean(y)-2*np.std(y))) #elimino righe oltre 2 sigma 
x=x[limit]
y=y[limit]



print(np.mean(y),np.std(y))

plt.scatter(x,y) #plotto il grafico 


popt, pcov = curve_fit(func, x, y,method='lm') #calcolo parametri della interpolazione lineare
plt.plot(x, func(x, *popt), 'k--', lw=1., label='y= %5.3f x + %5.3f'% tuple(popt)) #plotto la interpolazione linerare
std1=np.std(y-(func(x,*popt))) #calcolo la sd della differenza dalla interpolazione lineare
print(std1)  


plt.axis([0,5,-4,0]) #definisco i limiti del grafico
plt.colorbar()   
plt.legend()
plt.show()
