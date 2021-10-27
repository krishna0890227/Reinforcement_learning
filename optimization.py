
import numpy as np
import pandas as pd
import matplotlib.pyplot as pd
from scipy.optimize import minimize

def optimization_DES(Load_Pdata, PV_Pdata, Net_data):
   optimized_result=[]
   load_CI=1.96*np.std(Load_Pdata)/np.mean(Load_Pdata)
   PV_CI=1.96*np.std(PV_Pdata)/np.mean(PV_Pdata)
   Net_CI=1.96*np.std(Net_data)/np.mean(Net_data)
   load_lowerBounds=Load_Pdata-2*load_CI
   load_upperBounds=Load_Pdata+2*load_CI
   PV_lowerBounds=PV_Pdata-2*PV_CI
   PV_upperBounds=PV_Pdata + 2*PV_CI
   Net_upperBound=Net_data+2*Net_CI
   Net_lowerBound=Net_data-2*Net_CI

   print((Load_Pdata.shape, PV_Pdata.shape, Net_data.shape))

   for i in range (0, len(Load_Pdata)):
       L_low=load_lowerBounds[i]
       L_up=load_upperBounds[i]
       P_low=PV_lowerBounds[i]
       P_up=PV_upperBounds[i]
       N_Low=Net_lowerBound[i]
       N_up=Net_upperBound[i]

       def objective_fcn(X):
            x1=X[0]
            x2=X[1]
            x3=X[2]
            x4=X[3]
            result=x1-x2*x3-x4
            return result

       def constraint1(X):
           x2=X[2]
           return x2 -1

       # def constraint2(X):
       #     x2=X[1]
       #     return x2 - 1
       #
       # def constraint3(X):
       #     x2 = X[1]
       #     x4 = X[3]
       #     return x4 - x2

       x0=[Load_Pdata[i], 0, PV_Pdata[i], Net_data[i]] # initial guess

       x1_bounds=(L_low, L_up)
       x2_bounds=(0, 1)
       x3_bounds=(P_low, P_up)
       x4_bounds=(N_Low, N_up)
       bounds=[x1_bounds, x2_bounds, x3_bounds, x4_bounds]
       con1={'type': 'ineq', 'fun': constraint1}
       # con2 = {'type': 'ineq', 'fun': constraint2}
       # con3 = {'type': 'ineq', 'fun': constraint3}
       cons=[con1]
       sol=minimize(objective_fcn, x0, method='SLSQP', bounds=bounds, constraints=cons)
       output=sol.x
       result=output[0]-output[1]
       print(result)
       optimized_result.append(result)
   return optimized_result

