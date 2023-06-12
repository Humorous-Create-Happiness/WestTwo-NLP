import random

# 读取文本文件内容
file_path = "C:/Users/Lenovo/Desktop/py/102104126考核4/Summary.txt"  # 替换为实际的文件路径
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
i=500000           #读入行数
# 遮盖文本行中的每个字符
masked_lines = []

for line in lines:
    if len(line) > 420: continue
    if i > 0:
        masked_line = ""
        for char in line.strip():
            if char.strip() != "":
                if random.random() < 0.18:  # 遮盖概率
                    masked_line += "[MASK]"
                else:
                    masked_line += char
        masked_line = "[BEGIN]" + masked_line + "[END]"
        masked_lines.append(masked_line)
        i = i - 1
    else:
        break

# 打印遮盖后的文本行
#for masked_line in masked_lines:
    #print(masked_line)
#写入文件
with open("C:\\Users\Lenovo\Desktop\py\\102104126考核1\\234.txt", 'w', encoding='utf-8') as file:
    for masked_line in masked_lines:
        file.write(masked_line +'\n')





file_path = "C:/Users/Lenovo/Desktop/py/102104126考核4/Summary.txt"  # 替换为实际的文件路径
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
i = 500000  # 读入行数
alines=[]
for line in lines:
    if len(line) > 420: continue
    if i > 0:
        aline = ""
        for char in line.strip():
            if char.strip() != "":
                aline += char
        aline = "[BEGIN]" + aline + "[END]"
        alines.append(aline)
        i = i - 1

    else:
        break
with open("C:\\Users\Lenovo\Desktop\py\\102104126考核1\\233.txt", 'w', encoding='utf-8') as file:
    for aline in alines:
        file.write(aline + '\n')