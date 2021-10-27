
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def run_simulation(policy, initial_budget, initial_num_stocks, current_load, hist):
    budget=initial_budget
    num_stocks=initial_num_stocks
    future_value=0
    transition=list()
    current_result, final_result = [], []
    for i in range(0, (2000-hist-1)):
        current_state=np.asmatrix(np.hstack((current_load[i:i+hist], budget, num_stocks)))
        print(current_state.shape)
        Current_portfolio=budget + num_stocks*future_value
        action = policy.select_action(current_state, i)
        print(action)
        current_result.append(current_load[i+hist+1])
        future_value=float(current_load[i+hist])

        # update policy
        if action=='Sunny' and Current_portfolio>=future_value:
            budget -= future_value
            num_stocks += 1
        elif action=='Cloudy' and Current_portfolio<future_value:
            budget +=future_value
            num_stocks -= 1
        else:
            budget -= future_value
            num_stocks -= 1
        new_portfolio = budget + num_stocks * future_value
        reward = new_portfolio-Current_portfolio
        next_state = np.asmatrix(np.hstack((current_load[i+1:i+hist+1], budget, num_stocks)))
        print(next_state.shape)
        transition.append((current_state, action, reward, next_state))
        policy.update_q(current_state, action, reward, next_state)
        final_result.append(current_load[i+hist+1])

    portfolio=budget + num_stocks * future_value
    return portfolio, current_result, final_result

def run_simulations(policy, budget,num_stocks, current_load, hist):
    num_tries = 10
    final_portfolios = list()
    for  i in range(num_tries):
        final_portfolio, current_result, final_result = run_simulation(policy, budget, num_stocks, current_load, hist)
        final_portfolios.append(final_portfolio)
        print('Final portfolio: ${}'.format(final_portfolio))
    plt.title('Net Electricity Consumption (MW)')
    plt.plot(current_result, 'r-', label='Actual Net load')
    plt.plot(final_result, 'b-', label='Predicted Net Load')
    plt.xlabel('Time steps (t)')
    plt.legend(loc='lower left')
    plt.show()




