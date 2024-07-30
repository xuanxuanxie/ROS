# 读取文件并删除重复行的脚本
import random

input_file = '/home/invs/Desktop/实验/rda+occ/occ_ratio_2024-07-30_12-03-40.txt'
output_file = 'cut_sameline_occ111.txt'

# 读取文件
with open(input_file, 'r') as file:
    lines = file.readlines()

# 删除重复的行
unique_lines = list(set(lines))




# 按原顺序重新排序
unique_lines.sort(key=lines.index)

# 将结果写入新文件
with open(output_file, 'w') as file:
    file.writelines(unique_lines)

print(f"Cleaned file saved as {output_file}")
