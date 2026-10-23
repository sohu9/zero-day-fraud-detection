import matplotlib.pyplot as plt

# Set academic style for the report
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#333333'

# Create a high-resolution figure
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
fig.patch.set_facecolor('white')

# Data for the graph
metrics = ['TPR\n(Sensitivity)', 'TNR\n(Specificity)', 'FPR\n(False Alarms)', 'FNR\n(Missed Fraud)']
values = [95.92, 99.53, 0.47, 4.08]

# Colors: Green/Blue for good metrics, Orange/Red for error rates
colors = ['#27ae60', '#2980b9', '#f39c12', '#c0392b']

# Create the bars
bars = ax.bar(metrics, values, color=colors, edgecolor='black', linewidth=1.2, width=0.55)

# Formatting the graph
ax.set_ylim(0, 115) # Set Y-axis slightly higher to fit the text labels
ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax.set_title('Figure 4.3: Model Performance Rates Comparison', fontsize=14, fontweight='bold', pad=15)
ax.grid(axis='y', linestyle='--', alpha=0.6)

# Add exact percentage values on top of each bar
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f'{yval}%', 
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# Save the final image
plt.tight_layout()
plt.savefig('Figure_4_3_Metrics_Bar_Graph.png', dpi=300, bbox_inches='tight')
print("SUCCESS! Aapki image 'Figure_4_3_Metrics_Bar_Graph.png' ban gayi hai.")