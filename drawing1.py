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
marginal_prob_phn = np.random.rand(39)  # 确保数据中包含多个不同的值
marginal_prob_phn = np.exp(marginal_prob_phn)
marginal_prob_phn = marginal_prob_phn / marginal_prob_phn.sum()
entropy_phn = np.random.rand(39)  # 确保数据中包含多个不同的值
entropy_phn = np.exp(entropy_phn)

fig = plt.figure(figsize=(20, 15))
gs = fig.add_gridspec(3, 4, width_ratios=[5, 20, 4, 1], height_ratios=[3, 10, 2], 
                      hspace=0.1, 
                    #   vspace=0.1,
                      )

# 上方：unit 边际概率
ax_marginal = fig.add_subplot(gs[0, 1])
sns.barplot(x=np.arange(len(marginal_prob_unit)), y=marginal_prob_unit, ax=ax_marginal, color='blue')
ax_marginal.set_ylabel('Marginal Prob1')
ax_marginal.set_xticks([])
ax_marginal.set_xlabel('')
ax_marginal.set_xlim(-0.5, len(marginal_prob_unit) - 0.5, auto=None)  # 手动设置 x 轴范围

ax_newmargin = fig.add_subplot(gs[1, 0])
sns.barplot(y=np.arange(len(marginal_prob_phn)), 
            x=marginal_prob_phn, ax=ax_newmargin, color='blue',
            orient='h')
ax_newmargin.set_ylabel('Marginal Prob2')
ax_newmargin.set_ylim(-0.5, len(marginal_prob_phn) - 0.5, auto=None)  # 手动设置 x 轴范围
# ax_newmargin.set_yticks([])
# ax_newmargin.set_ylabel('')
ax_newmargin.invert_xaxis()  # 反转 y 轴方向
ax_newmargin.invert_yaxis()  # 反转 y 轴方向

# 中间：p_xy 热图
ax_heatmap = fig.add_subplot(gs[1, 1])
sns.heatmap(p_xy, ax=ax_heatmap, cmap='YlGnBu', cbar_ax=fig.add_subplot(gs[1, 3]), cbar_kws={'label': 'Probability'}, xticklabels=True)
ax_heatmap.set_ylabel('Phonemes2')
ax_heatmap.set_xlabel('Units1')
ax_heatmap.axvline(3, color='r', linewidth=1, linestyle="-")
ax_heatmap.axvline(7, color='r', linewidth=1, linestyle="-")

# 下方：unit 熵
ax_entropy = fig.add_subplot(gs[2, 1])
sns.barplot(x=np.arange(len(entropy_unit)), y=entropy_unit, ax=ax_entropy, color='red')
ax_entropy.set_ylabel('Entropy')
ax_entropy.set_xlabel('Units3')
ax_entropy.set_xlim(-0.5, len(entropy_unit) - 0.5, auto=None)  # 手动设置 x 轴范围
ax_entropy.invert_yaxis()  # 反转 x 轴方向

ax_newentropy = fig.add_subplot(gs[1, 2])
sns.barplot(y=np.arange(len(entropy_phn)), 
            x=entropy_phn, ax=ax_newentropy, color='red',
            orient='h')
ax_newentropy.set_ylabel('Entropy4')
ax_newentropy.yaxis.set_label_position('right')  # 移动 y 轴标签位置
# ax_newentropy.set_yticks([])
# ax_newentropy.set_ylabel('')
ax_newentropy.set_ylim(-0.5, len(entropy_phn) - 0.5, auto=None)  # 手动设置 x 轴范围
ax_newentropy.invert_yaxis()  # 反转 y 轴方向

# 仅为中间的热图设置单位标签
ax_heatmap.set_xticks(np.arange(len(marginal_prob_unit)))
ax_heatmap.set_xticklabels(np.arange(len(marginal_prob_unit)))

# 移除上方和下方的 x 轴标签
ax_marginal.set_xticks([])
ax_entropy.set_xticks([])
ax_newmargin.set_yticks([])
ax_newentropy.set_yticks([])
# 调整子图之间的间距
fig.subplots_adjust(hspace=0.5)  # 上方和中间的间距小一些
plt.subplots_adjust(bottom=0.15)  # 底部留出更多空间用于标签

# 为下方的熵图设置更大的间距
# gs.update(hspace=0.5)  # 中间和下方的间距大一些

# plt.tight_layout()
# plt.show()

import streamlit as st
st.pyplot(fig)
