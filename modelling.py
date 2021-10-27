import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
import Policy_update as PU
import Update_QFunction as QF
import Load_Profile_Analyis as LPA
from scipy.stats import pearsonr
import Neural_network as NN
import optimization as OP



# sample and their distribution
RawInput_data=pd.read_excel("data/LD_NET.xlsx")
RawInput_PVData=pd.read_excel("data/PV1_refine1_excel.xlsx")
PV_input1=RawInput_PVData.iloc[250:359, 1:25]
PV_input2=RawInput_PVData.iloc[:, 12:17]
print(PV_input2.describe())
PV_input=PV_input1.values
PV_input=PV_input.flatten()
print(len(PV_input))

RawNet_data=RawInput_data.iloc[:11630, :]
# data is ranging until 11631. So
print(RawNet_data.describe())
Net_data = RawNet_data.iloc[-115:-6, 1:].values
Net_data=Net_data.flatten()
print(Net_data.shape)
# load data Generation
print(Net_data.mean())
print('++++++ Load data Generation +++++++')
load_data=Net_data+PV_input
plt.plot(load_data, 'r-')
plt.show()

#
scaler, X_train, y_train, X_test, y_test, y_old, X_train_3D, X_test_3D= NN.data_preprocessing(Net_data)
scalerL, X_trainL, y_trainL, X_testL, y_testL, y_oldL, X_train_3DL, X_test_3DL= NN.data_preprocessing(load_data)
scalerP, X_trainP, y_trainP, X_testP, y_testP, y_oldP, X_train_3DP, X_test_3DP=NN.data_preprocessing(PV_input)
print((X_train.shape, y_train.shape))

Net_pred, net_mape, net_rmse=NN.BPNN_model(X_train, y_train, X_test, y_test, scaler)
print('Prediction passed ------- 1')
load_pred, load_mape, load_rmse=NN.BPNN_model(X_trainL, y_trainL, X_testL, y_testL, scalerL)
print('Prediction passed ------- 2')
PV_pred, PV_mape, PV_rmse=NN.BPNN_model(X_trainP, y_trainP, X_testP, y_testP, scalerP)
print('+-------------------------+')




Net_pred1=load_pred-PV_pred
Net_opt_result= OP.optimization_DES(load_pred, PV_pred, y_old)
net_mape=NN.nMAPE(y_test, Net_pred)
net_rmse=NN.nRMSE(y_test, Net_pred)
opt_mape=NN.nMAPE(y_test, Net_opt_result)
opt_rmse=NN.nRMSE(y_test, Net_opt_result)
print((net_mape, net_rmse, opt_mape, opt_rmse))
print(y_test)
print(Net_pred)
print(Net_opt_result)
plt.plot(y_test, 'r-', label='actual Net load data')
plt.plot(Net_pred, '.b-', label='predicted Net load data')
plt.plot(Net_opt_result, 'k--', label='Optimized Predicted data')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()






# seasonaltiy working
Net_data1=RawNet_data.iloc[-465:-100, 1:]
Net_data2=RawNet_data.iloc[-830:-465, 1:]
Net_data_input1=Net_data1.values
Net_data_input2=Net_data2.values
#LPA.plot_meanDifference(Net_data_input)
days_diff1, BTM_diff1, Seasonality_diff1 = LPA.daily_difference(Net_data_input1)
days_diff2, BTM_diff2, Seasonality_diff2 = LPA.daily_difference(Net_data_input2)
PV_diff1, BTM_diff3, Seasonality_diff3 = LPA.daily_difference(PV_input)

BTM_diff3=np.negative(BTM_diff3)
print(np.mean(BTM_diff1))
print(np.mean(BTM_diff2))
print(np.mean(Seasonality_diff1))
print(np.mean(Seasonality_diff2))
# BTM difference plot
plt.hist(BTM_diff3, 50, density=True, facecolor='g', alpha=0.75)
#plt.hist(Seasonality_diff2, 50, density=True, facecolor='r', alpha=0.75)
plt.legend(loc='upper left')
plt.show()

# seasonality difference plot
plt.plot(Seasonality_diff1, 'r-', label='This year')
plt.plot(Seasonality_diff2, 'b-', label='Last year')
plt.legend(loc='upper left')
plt.show()

#print(Net_data.head())
corr, _ = pearsonr(Seasonality_diff1, BTM_diff3)
print('Pearsons correlation: %.3f' % corr)


NetWork_data=Net_data.values
Network_data=NetWork_data.flatten()
print(len(Network_data))
LPA.periodically_evaluation(Network_data, 1)

# actions = ['Sunny','Rainy','Cloudy']
# hist = 100
# policy = QF.QLearningDecisionPolicy(actions, hist)
# budget = 10000
# num_stocks = 0
# PU.run_simulations(policy, budget, num_stocks, Network_data, hist )










# def param_analyis(sample):
#     mean = np.mean(sample)
#     std = np.std(sample)
#     min=np.min(sample)
#     max=np.max(sample)
#     dist = norm(mean, std)
#
#     # outcomes with range
#     values = [value for value in range(min, max)]
#     probability = [dist.pdf(value) for value in values]
#     # plot historgram and pdf
#
#     plt.hist(sample, bins=20, density=True)
#     plt.plot(values, probability, 'r-')
#     plt.show()
#
#     values = np.asarray([values])
#     param_mean = np.mean(values)
#     param_std = np.std(values)
#     print((param_mean, param_std))
#
#     return param_mean, param_std
#
#
# def nonparam_analyis(sample):
#     mean = np.mean(sample)
#     std = np.std(sample)
#     min = np.min(sample)
#     max = np.max(sample)
#
#
#     model = KernelDensity(bandwidth=2, kernel='gaussian')
#     sample = sample.reshape((len(sample), 1))
#     model.fit(sample)
#
#     # define values.
#     values = np.asarray([value for value in range(min, max)])
#     values = values.reshape((len(values), 1))
#     probability = model.score_samples(values)
#     prob_output = np.exp(probability)
#
#     # plot hist and PDF
#     plt.hist(sample, bins=20, density=True)
#     plt.plot(values[:], prob_output)
#     plt.show()
#
#     nonParam_mean = np.mean(values)
#     nonParam_std = np.std(values)
#     print((nonParam_mean, nonParam_std))
#
#     return nonParam_mean, nonParam_std
#
# param_analyis(Network_data)
# nonparam_analyis(Network_data)
#
#
#
# def parameteric():
#
#     sample=np.random.normal(50, 10, 1000)
#     mean=np.mean(sample)
#     std=np.std(sample)
#     dist=norm(mean, std)
#
#     # outcomes with range
#     values=[value for value in range(30, 70)]
#     probability = [dist.pdf(value) for value in values]
#     # plot historgram and pdf
#
#     plt.hist(sample, bins=10, density=True)
#     plt.plot(values, probability, 'r-')
#     plt.show()
#
#     values=np.asarray([values])
#     param_mean=np.mean(values)
#     param_std=np.std(values)
#
#     return param_mean, param_std
#
# def non_param():
#     sample1=np.random.normal(30, 5, 1000)
#     sample2=np.random.normal(50, 5, 2000)
#     sample=np.hstack((sample1, sample2))
#     # defining the density function
#     model=KernelDensity(bandwidth=3, kernel='gaussian')
#     sample=sample.reshape((len(sample), 1))
#     model.fit(sample)
#
#     # define values.
#     values=np.asarray([value for value in range(1, 60)])
#     values=values.reshape((len(values), 1))
#     probability=model.score_samples(values)
#     prob_output=np.exp(probability)
#
#     # plot hist and PDF
#     plt.hist(sample, bins=50, density=True)
#     plt.plot(values[:], prob_output)
#     plt.show()
#
#     nonParam_mean=np.mean(values)
#     nonParam_std=np.std(values)
#
#     return nonParam_mean, nonParam_std
#
# parameteric()
# non_param()



