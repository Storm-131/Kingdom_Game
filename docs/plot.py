import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Data for the bar plots
categories = ['Castle Strength', 'Resources', 'Soldiers']
values = [100, 50, 50]
colors = ['blue', 'green', 'red']

# Load a castle image for the background
castle_img = mpimg.imread('./background.jpg')  # Update this to your image's path

# Customizing the font style
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'serif',
    'font.serif': 'Times New Roman',  # This should be a font available on your system
    'font.style': 'italic'
})


fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Set the background image on both subplots
for ax in axes:
    # The extent would need to be adjusted based on the actual image dimensions
    ax.imshow(castle_img, aspect='auto', extent=[-1, 3, 0, 120], zorder=0, alpha=0.5)

# Plotting for Kingdom A
axes[0].bar(categories, values, color=colors, zorder=3)
axes[0].set_title('Kingdom A', fontsize=28, fontweight='bold')
axes[0].set_ylim(0, 120)

# Plotting for Kingdom B
axes[1].bar(categories, values, color=colors, zorder=3)
axes[1].set_title('Kingdom B', fontsize=28, fontweight='bold')
axes[1].set_ylim(0, 120)

# Main title and layout adjustment
# plt.suptitle('', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig('./example.png')
