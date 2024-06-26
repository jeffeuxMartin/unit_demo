import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 示例数据，替换为你的实际数据
p_xy = np.random.rand(39, 100)  # 替换为实际的 p_xy 数据
p_xy = p_xy / p_xy.sum()
marginal_prob_unit = np.random.rand(100)  # 确保数据中包含多个不同的值
marginal_prob_unit = np.exp(marginal_prob_unit)
marginal_prob_unit = marginal_prob_unit / marginal_prob_unit.sum()
entropy_unit = np.random.rand(100)  # 确保数据中包含多个不同的值
entropy_unit = np.exp(entropy_unit)

fig = plt.figure(figsize=(20, 15))
gs = fig.add_gridspec(3, 2, width_ratios=[20, 1], height_ratios=[1, 5, 1], hspace=0.05)

# 上方：unit 边际概率
ax_marginal = fig.add_subplot(gs[0, 0])
sns.barplot(x=np.arange(len(marginal_prob_unit)), y=marginal_prob_unit, ax=ax_marginal, color='blue')
ax_marginal.set_ylabel('Marginal Prob')
ax_marginal.set_xticks([])
ax_marginal.set_xlabel('')
ax_marginal.set_xlim(-0.5*2, len(marginal_prob_unit) + 0.0, auto=None)  # 手动设置 x 轴范围

# 中间：p_xy 热图
ax_heatmap = fig.add_subplot(gs[1, 0], sharex=ax_marginal)
sns.heatmap(p_xy, ax=ax_heatmap, cmap='YlGnBu', cbar_ax=fig.add_subplot(gs[1, 1]), cbar_kws={'label': 'Probability'}, xticklabels=False)

ax_heatmap.set_ylabel('Phonemes')
ax_heatmap.set_xlabel('Units')

# 下方：unit 熵
ax_entropy = fig.add_subplot(gs[2, 0], sharex=ax_marginal)
sns.barplot(x=np.arange(len(entropy_unit)), y=entropy_unit, ax=ax_entropy, color='red')
ax_entropy.set_ylabel('Entropy')
ax_entropy.set_xlabel('Units')
ax_entropy.set_xlim(-0.5*2, len(entropy_unit) + 0.0, auto=None)  # 手动设置 x 轴范围

# 为中间的热图设置单位标签
ax_heatmap.set_xticks(np.arange(len(marginal_prob_unit)))
ax_heatmap.set_xticklabels(np.arange(len(marginal_prob_unit)))

# 移除上方和下方的 x 轴标签
ax_marginal.set_xticklabels([])
ax_entropy.set_xticklabels([])
ax_heatmap.set_xticklabels(np.arange(len(marginal_prob_unit)))

plt.tight_layout()
plt.show()
import streamlit as st
st.pyplot(fig)
