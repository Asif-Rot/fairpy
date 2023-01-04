import matplotlib.pyplot as plt
import numpy as np
from fairpy.agentlist import AgentList
from Algorithms_numba import optimal_envy_free, optimal_envy_free1
import time


def plot_graph(n):
    """
    This function will plot a graph from all the calculations
    """
    labels = []
    algo = []
    algo_times = []
    algo_numba = []
    algo_numba_times = []
    ex3 = AgentList({"Alice": {'1': 250, '2': 750}, "Bob": {'1': 250, '2': 750}})
    for j in range(1, n):
        optimal_envy_free(ex3, 1000, {'Alice': 600, 'Bob': 500})

        num_agents = 10 * j
        num_rooms = 10 * j
        agents = AgentList({f"Agent{i}": {f"Room{j}": (i + j) * 10 for j in range(num_rooms)} for i in range(num_agents)})
        rent = num_agents * num_rooms * 10
        budgets = {f"Agent{i}": (i + 1) * 100 for i in range(num_agents)}
        labels.append(10 * j)

        start_time = time.perf_counter()
        optimal_envy_free1(agents, rent, budgets)
        end_time = time.perf_counter()
        time_result = end_time - start_time
        time_result_str = "{:.4f}".format(time_result)
        algo.append(time_result)
        algo_times.append(time_result_str)

        start_time = time.perf_counter()
        optimal_envy_free(agents, rent, budgets)
        end_time = time.perf_counter()
        time_result = end_time - start_time
        time_result_str = "{:.4f}".format(time_result)
        algo_numba.append(time_result)
        algo_numba_times.append(time_result_str)

    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, algo, width, label='algo')
    rects2 = ax.bar(x + width / 2, algo_numba, width, label='algo_numba')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Time in seconds')
    ax.set_title('Compare between algo and algo_numba')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.legend()

    for rect, label in zip(rects1, algo_times):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height, label,
                ha='center', va='bottom', rotation=270)

    for rect, label in zip(rects2, algo_numba_times):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height, label,
                ha='center', va='bottom', rotation=270)

    fig.tight_layout()

    plt.show()


if __name__ == '__main__':
    plot_graph(11)
