import matplotlib.pyplot as plt
import numpy as np

# 文件路径
data_points_file = "initial_data_points.txt"
cluster_assignments_file = "cluster_assignments.txt"
final_centroids_file = "final_centroids.txt"

# 读取初始数据点
data_points = np.loadtxt(data_points_file)

# 读取簇分配结果
cluster_assignments = np.loadtxt(cluster_assignments_file, dtype=int)

# 读取最终簇中心
final_centroids = np.loadtxt(final_centroids_file)

# 可视化
plt.figure(figsize=(10, 8))

# 簇的颜色
colors = ['red', 'blue', 'green']

# 绘制每个簇的数据点
for cluster_id in range(len(final_centroids)):
    cluster_points = data_points[cluster_assignments == cluster_id]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=10, c=colors[cluster_id], label=f'Cluster {cluster_id}')

# 绘制簇中心
plt.scatter(final_centroids[:, 0], final_centroids[:, 1], marker='X', c='black', s=150, label='Centroids')

# 图形美化
plt.title('K-Means Clustering Visualization')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend()
plt.grid(True)

# 显示图形
plt.show()