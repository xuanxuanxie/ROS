import numpy as np
import matplotlib.pyplot as plt

# 示例数据：不同算法在多次运行中的目标点数
algorithm_A = np.array([120, 150, 200, 250, 300, 350, 400])
algorithm_B = np.array([100, 180, 220, 280, 310, 360, 410])
algorithm_C = np.array([130, 160, 210, 260, 320, 370, 420])

# 计算CDF函数
def compute_cdf(data):
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, cdf

# 计算各算法的CDF
sorted_A, cdf_A = compute_cdf(algorithm_A)
sorted_B, cdf_B = compute_cdf(algorithm_B)
sorted_C, cdf_C = compute_cdf(algorithm_C)

# 绘制CDF
plt.figure(figsize=(8, 6))
plt.plot(sorted_A, cdf_A, label='Algorithm A', marker='o')
plt.plot(sorted_B, cdf_B, label='Algorithm B', marker='o')
plt.plot(sorted_C, cdf_C, label='Algorithm C', marker='o')

# 添加标题和标签
plt.title('CDF of Object Point Counts for Different Algorithms')
plt.xlabel('Object Point Count')
plt.ylabel('Cumulative Distribution Function (CDF)')
plt.legend()
plt.grid(True)

# 显示图形
plt.show()
