import matplotlib.pyplot as plt
import numpy as np

# Set academic style
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#333333'

# Create a high-resolution figure
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
fig.patch.set_facecolor('white')

# Data for XAI Feature Residuals (Example of a Zero-Day Fraud Transaction)
features = ['Geographic Distance\n(Location Jump)', 'IP Risk Score\n(VPN/Tor)', 'Spending Velocity', 'Device Signature\n(New OS/Hardware)', 'Time Delta\n(Midnight Burst)']
# These values represent how much each feature contributed to the high MSE
importance = [0.85, 0.62, 0.45, 0.35, 0.20] 

# Colors: Red/Orange tones to represent risk/anomaly contribution
colors = ['#c0392b', '#e74c3c', '#e67e22', '#f39c12', '#f1c40f']

# Create horizontal bars
bars = ax.barh(features, importance, color=colors, edgecolor='black', height=0.6)

# Formatting the graph
ax.set_xlabel('Reconstruction Error (MSE) Contribution', fontsize=12, fontweight='bold')
ax.set_title('Figure 4.4: XAI Feature Residuals for Flagged Anomaly', fontsize=14, fontweight='bold', pad=15)
ax.invert_yaxis() # Highest contribution at the top
ax.grid(axis='x', linestyle='--', alpha=0.6)

# Add value text on the bars
for bar in bars:
    width = bar.get_width()
    ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, f'+{width:.2f} MSE', 
            ha='left', va='center', fontsize=11, fontweight='bold', color='#333333')

# Draw a vertical line showing a reference threshold limit conceptually
ax.axvline(x=0.5, color='red', linestyle=':', linewidth=2, alpha=0.5, label='High Risk Zone')
ax.legend(loc='lower right')

# Save the final image
plt.tight_layout()
plt.savefig('Figure_4_4_XAI_Residuals.png', dpi=300, bbox_inches='tight')
print("SUCCESS! Aapki image 'Figure_4_4_XAI_Residuals.png' ban gayi hai.")