#---------------------------------------------------------*\
# Title: Plotting the Results
# Author: 
#---------------------------------------------------------*/
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy.stats import linregress
import matplotlib as mpl
from scipy.stats import norm

def plot_evaluation(action_freqs, win_rates, average_rewards, item_names={'0': 'Gather Resources', '1': 'Train Soldiers', '2': 'Fortify Castle', '3': 'Attack'}, save_path="./out/combined_plot.pdf"):
    """
    Combines plotting of item frequencies, win rates, average rewards, and a pie chart of won games.
    
    :param action_freqs: Array of dictionaries with frequencies of items for the first subplot.
    :param win_rates: List of win rates for the second subplot.
    :param average_rewards: List of average rewards for the third subplot.
    :param item_names: Optional dictionary to map original item names to custom names for the first subplot.
    :param save_path: Path to save the combined plot.
    """
    
    fig = plt.figure(figsize=(12, 10))
    plt.suptitle("Evaluation Metrics Overview", fontsize=20, fontweight="bold")  # Add your supertitle here

    # Set font properties globally for bold text
    mpl.rcParams['font.weight'] = 'bold'
    mpl.rcParams['axes.labelweight'] = 'bold'
    mpl.rcParams['axes.titleweight'] = 'bold'

    #---------------------------------------------------------*/
    ## 1) Item Frequencies
    #---------------------------------------------------------*/
    plt.subplot(2, 2, 1)
    
     # Initialize a dictionary to accumulate frequencies
    frequencies = {item_names[str(i)]: [] for i in range(len(item_names))}    
    
    # Accumulate frequencies for each item in each data point
    for d in action_freqs:
        for key, value in item_names.items():
            frequencies[value].append(d.get(value, 0))
           
     # Plotting the data for each item 
    for item, freq in frequencies.items():
        plt.plot(freq, label=item)
        
    plt.xlabel('Data Point Index')
    plt.ylabel('Frequency')
    plt.title('Item Frequencies')
    plt.legend()
    plt.grid(True)
  
    #---------------------------------------------------------*/
    # 2) Win Rate Distribution
    #---------------------------------------------------------*/
    plt.subplot(2, 2, 2)
    colors = ['yellow' if rate == 0.5 else 'green' if rate > 0.5 else 'red' for rate in win_rates]
    plt.bar(range(len(win_rates)), [1] * len(win_rates), color=colors)
    plt.title("Win Rate Distribution")
    plt.xlabel("Episode")
    plt.yticks([])

    # Calculating the overall win rate percentage
    wins = sum(1 for rate in win_rates if rate > 0.5)
    losses = len(win_rates) - wins
    overall_win_rate = (wins / len(win_rates)) * 100
    plt.text(0.95, 0.85, f'Win Rate: {overall_win_rate:.2f}%', transform=plt.gca().transAxes, 
             horizontalalignment='right', color='black', bbox=dict(facecolor='white', alpha=0.75), fontsize=12)

    #---------------------------------------------------------*/
    # 3) Average Reward Over Time
    #---------------------------------------------------------*/
    plt.subplot(2, 2, 3)
    
    plt.plot(average_rewards)
    plt.title("Average Reward Over Time")
    plt.xlabel("Episodes (x100)")
    plt.ylabel("Average Reward")
    x = np.arange(len(average_rewards))
    slope, intercept, _, _, _ = linregress(x, average_rewards)
    plt.plot(x, intercept + slope * x, label="Trend Line", color='red')

    #---------------------------------------------------------*/
    # 4) Won Games
    #---------------------------------------------------------*/
    plt.subplot(2, 2, 4)
    
    episode_lengths = [sum(d.values()) for d in action_freqs]
    mean, std = np.mean(episode_lengths), np.std(episode_lengths)

    # Generate points on the x axis between -3 and +3 standard deviations of the mean
    x = np.linspace(mean - 3*std, mean + 3*std, 100)

    # Plot normal distribution with mean and std
    plt.plot(x, norm.pdf(x, mean, std), label='Normal Distribution')

    # Plot histogram of episode lengths
    plt.hist(episode_lengths, bins=30, density=True, alpha=0.6, color='g', label='Histogram')

    plt.title("Episode Length Distribution")
    plt.xlabel("Episode Length")
    plt.ylabel("Density")
    plt.legend()
    
    #---------------------------------------------------------*/
    # General Stuff
    #---------------------------------------------------------*/
    
    # Add the background image to the figure
    background = fig.add_axes([0, 0, 1, 1], zorder=-1)
    background.imshow(mpimg.imread('./docs/background.jpg'), aspect='auto', extent=[-1, 3, 0, 120], alpha=0.1)
    background.set_xticks([])
    background.set_yticks([])
    background.set_frame_on(False)
    
    plt.savefig(save_path)
    plt.show()
    plt.close('all')

# Example usage:
# combined_plot(data, win_rates, average_rewards)


#-------------------------Notes-----------------------------------------------*\
# 
#-----------------------------------------------------------------------------*\