import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import re
import tensorwatch as tw



# 检查是否有可用的GPU设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print (666)
# 定义模型参数
embed_dim = 512    # 词向量维度/2
num_heads = 4       #多头数
hidden_dim = 256   # 隐藏层维度/2
num_layers = 4      # 网络层数
max_seq_len = 700  #最大句子长度（仅做标记）
#在retrain中/1.4才是输入最大列


# 定义数据集
dataset = [
    "[BEGIN]事实证明 8M 参[MASK]就能做[MASK]差强人意的模型出来。[END]",
    "[BEGIN]这是另一个例子，你可以根据需要添加更多的句子。[END]"
]
with open('234.txt', 'r', encoding='utf-8') as f:#与训练时一致
    lines = f.readlines()
    for line in lines:
        dataset.append(line.strip())
print('finish!')

# 构建词汇表
vocab = set()
for sentence in dataset:
    vocab.update(list(sentence))
vocab = list(vocab)
vocab_size = 6062###################################################################################len(vocab)   #词汇表大小
print("词汇表大小:")
print(vocab_size)         #输出vocab_size的大小
char2idx = {c: i for i, c in enumerate(vocab)}
idx2char = {i: c for i, c in enumerate(vocab)}

# 添加未知字符
unknown_token = "[UNK]"
vocab.append(unknown_token)
char2idx[unknown_token] = len(vocab) - 1
idx2char[len(vocab) - 1] = unknown_token

input_seqs = []
target_seqs = []
for sentence in dataset:
    input_text = sentence.replace("[MASK]", "")
    target_text = sentence.replace("[MASK]", "")
    input_seq = [char2idx.get(c, char2idx[unknown_token]) for c in input_text]
    target_seq = [char2idx.get(c, char2idx[unknown_token]) for c in target_text]
    input_seqs.append(input_seq)
    target_seqs.append(target_seq)

# 定义模型和其他组件
class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, dropout=0.5):#加大dropout层可以减小过拟合的概率
        super(Transformer, self).__init__()

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        x = self.positional_encoding(x)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        x = self.fc(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_len=700):#######################################################修改最大句子长度
        super(PositionalEncoding, self).__init__()

        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))

        self.positional_encoding = nn.Parameter(torch.zeros(max_seq_len, embed_dim))
        self.positional_encoding.data[:, 0::2] = torch.sin(position * div_term)
        self.positional_encoding.data[:, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        x = x + self.positional_encoding[:x.size(1), :]
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout):
        super(TransformerBlock, self).__init__()

        self.attention = MultiheadAttention(embed_dim, num_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        residual = x
        x = self.attention(x)
        x = self.dropout1(x)
        x = self.norm1(residual + x)
        residual = x
        x = self.fc(x)
        x = self.dropout2(x)
        x = self.norm2(residual + x)
        return x

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiheadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = torch.softmax(scores, dim=-1)

        x = torch.matmul(attention_weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        #x = self.fc(x)
        x =self.fc(x.view(-1, self.embed_dim))
        return x

# 创建并初始化模型
#model = Transformer(vocab_size, embed_dim, num_heads, hidden_dim, num_layers).to(device)
#optimizer = optim.Adam(model.parameters(), lr=0.001)
#criterion = nn.CrossEntropyLoss()

# 将数据转换为Tensor
input_tensors = [torch.tensor(seq).to(device) for seq in input_seqs]
target_tensors = [torch.tensor(seq).to(device) for seq in target_seqs]

# 训练模型(没有训练)
num_epochs = 2
batch_size = 128          #批次大小
total_batches = len(input_tensors)


#正确的文本
correct_texts=["[BEGIN]事实证明 8M 参数就能做出差强人意的模型出来。[END]","[BEGIN]这是另一个例子，你可以根据需要添加更多的句子。[END]"]
with open('233.txt', 'r', encoding='utf-8') as f:#与训练时一致
    lines = f.readlines()
    for line in lines:
        correct_texts.append(line.strip())
print('finish!')


# 加载模型
model = Transformer(vocab_size, embed_dim, num_heads, hidden_dim, num_layers).to(device)
model.load_state_dict(torch.load("model.pth"))
model.eval()


# 使用模型进行预测并保存于output_texts
input_texts = ["[BEGIN]事实证明 8M 参[MASK]就能做[MASK]差强人意的模型出来。[END]", "[BEGIN]这是另一[MASK]例子，你可以根据需要添加更多的句子。[END]"]
with open('234.txt', 'r', encoding='utf-8') as f:#与训练时一致
    lines = f.readlines()
    for line in lines:
        input_texts.append(line.strip())
print('finish!')

correct_count = 0
total_count = 0


# 重新训练与反馈训练
def train_model(model, optimizer, criterion, input_tensors, target_tensors, correct_texts, num_epochs, batch_size):
    total_batches = len(input_tensors)

    train_size=134217728
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        '''for step, (input_tensor, target_tensor) in enumerate(zip(input_tensors, target_tensors), 1):
            optimizer.zero_grad()
            output_tensor = model(input_tensor.unsqueeze(0))
            loss = criterion(output_tensor.squeeze(0)[:-1, :], target_tensor[:-1])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # 打印每个步骤的进度
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{step}/{total_batches}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / total_batches
        print(f"Avg Loss: {avg_loss:.4f}")'''



        # 将正确文本与生成的输出文本进行比较并计算损失
        weizhi=0          #位置记录
        input_texts = ["[BEGIN]事实证明 8M 参[MASK]就能做[MASK]差强人意的模型出来。[END]", "[BEGIN]这是另一[MASK]例子，你可以根据需要添加更多的句子。[END]"]
        with open('234.txt', 'r', encoding='utf-8') as f:  # 与训练时一致
            lines = f.readlines()
            for line in lines:
                input_texts.append(line.strip())
        print('finish  input_texts!')
        for input_text in input_texts:

            input_seq = [char2idx.get(c, char2idx[unknown_token]) for c in input_text]
            input_tensor = torch.tensor(input_seq).unsqueeze(0).to(device)
            output_tensor = model(input_tensor)
            output_seq = torch.argmax(output_tensor.squeeze(0), dim=-1)

            output_text = input_text
            mask_count = 0
            for i in range(len(output_text)):
                if output_text[i:i + len('[MASK]')] == '[MASK]':
                    mask_count += 1
                    if mask_count <= len(output_seq):

                        predicted_char = idx2char[output_seq[mask_count - 1].item()]
                        predicted_CHAR = predicted_char
                        while re.match(r'[a-zA-Z0-9\[\]\s]', predicted_char) or predicted_char in ['[MASK]', '[BEGIN]', '[END]'] or predicted_char == '' or predicted_char == predicted_CHAR:
                            new_output_seq = torch.argmax(output_tensor.squeeze(0), dim=-1)
                            new_output_seq[mask_count - 1] = torch.randint(low=0, high=vocab_size, size=(1,)).item()
                            predicted_char = idx2char[new_output_seq[mask_count - 1].item()]
                            if predicted_CHAR == predicted_char:
                                predicted_char = idx2char[new_output_seq[mask_count - 1].item()]

                        output_text = output_text[:i] + predicted_char + output_text[i + len('[MASK]'):]
                    else:
                        output_text = output_text[:i] + '' + output_text[i + len('[MASK]'):]

            #print("Input Text:", input_text)
            #print("Output Text:", output_text)

            correct_text=correct_texts[weizhi]
            #print ("Correct Text",correct_text)
            weizhi = weizhi + 1             #位置更新

            # 将正确文本与生成的输出文本转换为Tensor
            correct_tensor = torch.tensor([char2idx.get(c, char2idx[unknown_token]) for c in correct_text]).unsqueeze(0).to(device)
            output_tensor = torch.tensor([char2idx.get(c, char2idx[unknown_token]) for c in output_text]).unsqueeze(0).to(device)
            output_tensor = output_tensor.float()
            correct_tensor = correct_tensor.float()               #加入float类型
            output_tensor = output_tensor.requires_grad_()
            correct_tensor = correct_tensor.requires_grad_()      #加入grad要求避免报错

            # 计算损失并进行反向传播和优化
            loss = criterion(output_tensor.squeeze(0)[:-1], correct_tensor.squeeze(0)[:-1])
            #loss = criterion(output_tensor.squeeze(0)[:-1].view(-1, vocab_size), correct_tensor.squeeze(0)[:-1].view(-1, vocab_size))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()/train_size



            if weizhi %5000 == 1:
                print(f"Epoch [{epoch + 1}/{num_epochs}],Loss:{loss.item()/train_size:4f}")

        avg_loss = total_loss / total_batches
        print(f"Avg Loss: {avg_loss:.4f}")
            #print("-----------------------------------------")




# 创建并初始化模型
model = Transformer(vocab_size, embed_dim, num_heads, hidden_dim, num_layers).to(device)
model.load_state_dict(torch.load("model.pth"))     #运行环境与之前是否相同，在默认情况下是True
model.eval()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 将数据转换为Tensor
input_tensors = [torch.tensor(seq).to(device) for seq in input_seqs]
target_tensors = [torch.tensor(seq).to(device) for seq in target_seqs]

# 训练模型
num_epochs = 1
batch_size = 256
train_model(model, optimizer, criterion, input_tensors, target_tensors, correct_texts, num_epochs, batch_size)

# 保存模型
torch.save(model.state_dict(), "model.pth")


