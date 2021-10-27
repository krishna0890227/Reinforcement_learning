import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def periodically_evaluation (Net_data, period):
    quaters=period*24
    Yearly_mean, Yearly_var=[], []
    for i in range (0, int(len(Net_data)/quaters)):
        #print(i)
        Year_data= Net_data[i*quaters:(i+1)*quaters]
        year_mean=np.mean(Year_data)
        year_var=np.std(Year_data)
        Yearly_mean.append(year_mean)
        Yearly_var.append(year_var)

    print(len(Yearly_mean))

    yearMean_result=np.mean(Yearly_mean)
    yearvar_result=np.mean(Yearly_var)
    print(yearMean_result)
    print(yearvar_result)
    plt.plot(Yearly_mean, 'r-', label='Daily Mean')
    plt.plot(Yearly_var,  'b-', label='Daily Var')
    plt.legend(loc='upper left')
    plt.ylim([0, 80000])
    plt.show()


def plot_meanDifference(Net_data):
    yearlyDays=365
    yearlyMeanProfile=[]
    print(int(len(Net_data) / yearlyDays))
    for i in range(0, int(len(Net_data)/yearlyDays)):
        print('I am here')
        work_data=Net_data[i*yearlyDays:(i+1)*yearlyDays, :]
        mean_profile=np.mean(work_data, axis=0)
        yearlyMeanProfile.append(mean_profile)

    print(yearlyMeanProfile[0].shape)
    plt.plot(yearlyMeanProfile[0], 'r-', label='Last year')
    plt.plot(yearlyMeanProfile[1], '.b-', label='This year')
    plt.legend(loc='upper left')
    plt.ylim([50000, 70000])
    plt.show()

def daily_difference(Net_data):
    days_diff=[]
    BTM_difference=[]
    Seasonlity_difference=[]
    for i in range(0, len(Net_data)-1):
        day_profile=Net_data[i, :]
        # mean_day=np.mean(day_profile)
        mean_diff=day_profile
        seasonality_index=[0, 1, 2, 3, 4, 5, 19, 20, 21, 22, 23]
        BTM_index=[6,7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        daily_diff_sum=0
        BTM_impact_sum=0
        Seasonality_impact_sum=0
        seasonality_matrix=mean_diff[[seasonality_index]]
        Seasonality_impact_sum=np.max(seasonality_matrix)
        BTM_matrix=mean_diff[[BTM_index]]
        BTM_impact_sum=np.max(BTM_matrix)
        daily_diff_sum=np.mean(mean_diff)
        # print(daily_diff_sum)
        # print(BTM_impact_sum)
        # print(Seasonality_impact_sum)
        # #print('++++++++++++++++++++++++++++++')

        days_diff.append(daily_diff_sum)
        BTM_difference.append(BTM_impact_sum)
        Seasonlity_difference.append(Seasonality_impact_sum)
    return days_diff, BTM_difference, Seasonlity_difference














# yearly Change Table
# x_value=[55868.66, 56546.42, 57359.57, 59076.61, 61009.83, 60906.47]
# y_value=[4688.25, 4654.33, 5085.54, 5218.80, 5288.28, 4977.79]
# for i in range(0, len(x_value)-1):
#     present=y_value[i]
#     future=y_value[i+1]
#     change_per=((future-present)/future)*100
#     print(change_per)