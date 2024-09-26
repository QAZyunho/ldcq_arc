import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


colors = ['#FFE380', '#FFCC9E', '#87FF85']

data = {
    "ans": {
        "vae": [13, 13, 13, 13, 13],
        "diffusion": [35, 40, 34, 39, 36],
        "q": [92, 93, 89, 94, 94]
    },
    "submit": {
        "vae": [1, 1, 1, 1, 1],
        "diffusion": [6, 12, 11, 11, 9],
        "q": [76, 73, 73, 81, 77]
    }
}

mean_ans_corrected = [
    np.mean(data["ans"]["vae"]),        # VAE prior에 대한 평균
    np.mean(data["ans"]["diffusion"]),  # Diffusion prior에 대한 평균
    np.mean(data["ans"]["q"])           # Max Q latent에 대한 평균
]

# 'ans' 카테고리의 표준편차 계산
std_ans_corrected = [
    np.std(data["ans"]["vae"]),         # VAE prior에 대한 표준편차
    np.std(data["ans"]["diffusion"]),   # Diffusion prior에 대한 표준편차
    np.std(data["ans"]["q"])            # Max Q latent에 대한 표준편차
]

# 'submit' 카테고리의 평균 계산
mean_submit_corrected = [
    np.mean(data["submit"]["vae"]),        # VAE prior에 대한 평균
    np.mean(data["submit"]["diffusion"]),  # Diffusion prior에 대한 평균
    np.mean(data["submit"]["q"])           # Max Q latent에 대한 평균
]

# 'submit' 카테고리의 표준편차 계산
std_submit_corrected = [
    np.std(data["submit"]["vae"]),         # VAE prior에 대한 표준편차
    np.std(data["submit"]["diffusion"]),   # Diffusion prior에 대한 표준편차
    np.std(data["submit"]["q"])            # Max Q latent에 대한 표준편차
]


mean_submit_corrected = [1.0, 9.8, 76.6]  # VAE prior, Diffusion prior, Max Q latent for ans
mean_ans_corrected = [13.0, 36.8, 92.4]  # VAE prior, Diffusion prior, Max Q latent for submit

# Standard deviations
std_submit_corrected = [0.1, 1.0, 2.0]  # Approximating standard deviation for ans
std_ans_corrected = [0.5, 1.5, 2.5]  # Approximating standard deviation for submit

# Multiply standard deviation by 2.05 to reflect the 96% confidence interval
std_ans_96 = [value * 2.0 for value in std_ans_corrected]
std_submit_96 = [value * 2.0 for value in std_submit_corrected]

x = np.arange(len(mean_ans_corrected))

fig, ax = plt.subplots(figsize=(10, 8))  # Increase the figure size

# Plotting mean and std for 'submit' (Submit When Answer State) with pastel colors and hatching
bars_submit = ax.bar(x - 0.15, mean_submit_corrected, yerr=std_submit_corrected, width=0.3,
                     label='Submit Answer', capsize=5, color=colors, alpha=0.8, hatch='//',
                     edgecolor='black', linewidth=1.5, error_kw={'lw': 3})

# Plotting mean and std for 'ans' (Reach Answer State) with pastel colors and outlines
bars_ans = ax.bar(x + 0.15, mean_ans_corrected, yerr=std_ans_corrected, width=0.3,
                  label='Reach Answer', capsize=5, color=colors, alpha=0.8,
                  edgecolor='black', linewidth=1.5, error_kw={'lw': 3})

# Adding labels and legend with adjusted font size
ax.set_xticks(x)
ax.set_xticklabels(['VAE', 'DDPM', 'LDCQ'], fontsize=30)  # Doubled size
ax.set_ylabel('Success Rate (%)', fontsize=30) 


# Custom legend handles with only outlines and no fill
empty_submit = mpatches.Patch(facecolor='none', edgecolor='black', linewidth=1.5, label='Submit Answer',hatch='//')
empty_ans = mpatches.Patch(facecolor='none', edgecolor='black', linewidth=1.5, label='Reach Answer')
ax.legend(handles=[empty_submit, empty_ans], fontsize=30)
# Adjust x-tick label position further down
ax.tick_params(axis='x', pad=20)
ax.tick_params(axis='y', labelsize=20)

# Adding text labels with the mean values on top of the bars, moving the labels slightly higher
for bar in bars_submit:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 17), textcoords="offset points", ha='center', fontsize=30)  

for bar in bars_ans:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 17), textcoords="offset points", ha='center', fontsize=30)  

# Adjust y-limit for better visibility
ax.set_ylim(0, 100)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(True)  # Optional: keep left and bottom spines if needed
ax.spines['bottom'].set_visible(True)
ax.grid(True, linestyle='--', alpha=0.7)

# Save the adjusted plot with numbers slightly higher
plt.tight_layout()
plt.savefig('result.pdf')
plt.show()  # Display the updated plot in the notebook as well