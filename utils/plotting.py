#---------------------------------------------------------*\
# Title: Plotting the Results
# Author: 
#---------------------------------------------------------*/
#!/usr/bin/env python3

from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import linregress
    
def plot_item_frequencies(data, item_names={'0': 'Gather Resources', '1': 'Train Soldiers', '2': 'Fortify Castle', '3': 'Attack'}):
    """
    Plots line diagrams for frequencies of items in the provided array of dictionaries.

    :param data: Array of dictionaries with frequencies of items.
    :param item_names: Optional dictionary to map original item names to custom names.
    """
    # Initialize a dictionary to accumulate frequencies
    frequencies = {item_names[str(i)]: [] for i in range(len(item_names))}

    # Accumulate frequencies for each item in each data point
    for d in data:
        for key, value in item_names.items():
            frequencies[value].append(d.get(value, 0))

    # Creating the plot
    plt.figure(figsize=(10, 6))

    # Plotting the data for each item
    for item, freq in frequencies.items():
        plt.plot(freq, label=item)

    # Setting plot details
    plt.ylim(bottom=0)
    plt.xlabel('Data Point Index')
    plt.ylabel('Frequency')
    plt.title('Line Plot of Item Frequencies')
    plt.legend()
    plt.grid(True)
    plt.savefig('./out/Freqs.pdf')
    plt.show()
    plt.close('all')

def plot_results(win_rates, average_rewards, save_path="./out/plot.pdf"):
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    colors = ['yellow' if rate == 0.5 else 'green' if rate > 0.5 else 'red' for rate in win_rates]
    bars = plt.bar(range(len(win_rates)), [1] * len(win_rates), color=colors)
    plt.title("Win Rate Distribution")
    plt.xlabel("Episode")
    plt.yticks([])

    # Calculating the overall win rate percentage
    overall_win_rate = sum(win_rates) / len(win_rates) * 100
    plt.text(0.95, 0.85, f'Win Rate: {overall_win_rate:.2f}%', transform=plt.gca().transAxes,
         horizontalalignment='right', color='black',  bbox=dict(facecolor='white', alpha=0.75), fontsize=12)

    plt.subplot(1, 2, 2)
    
    # Linear Regression for the second subplot
    plt.plot(average_rewards)
    plt.title("Average Reward Over Time")
    plt.xlabel("Episodes (x100)")
    plt.ylabel("Average Reward")
    plt.tight_layout()
    
    x = np.arange(len(average_rewards))
    slope, intercept, _, _, _ = linregress(x, average_rewards)
    plt.plot(x, intercept + slope * x, label="Trend Line", color='red')

    plt.savefig(save_path)  # Save the plot to the specified path
    plt.show()
    plt.close('all')


#-------------------------Notes-----------------------------------------------*\
# 
#-----------------------------------------------------------------------------*\
