## PythonAI第四轮-------NLP模型训练挖空句子填写心得 ##

需求分析：

1.基于 Attention 机制搭建一个 Transform 模型，模型可以参考 pytorch 的教程、参考开源

模型如 Bert、GPT 等

2.训练时采用无监督学习，预测掩蔽字的方法训练：

3.整个模型不能超过 50M（五千万）

4.完成句子挖空填写的任务



### 一.transformer模型定义 ###

#### 模型参数定义

```py
# 定义模型参数
embed_dim = 128      # 词向量维度
num_heads = 2       # 多头数
hidden_dim = 128     # 隐藏层维度
num_layers = 4      # 网络层数
max_seq_len = 700   # 最大句子长度（仅做标记）允许的最大句子输入长度
```



#### 1.`transformer`(模型类)

```py
class Transformer(nn.Module):

    ##__init__ 方法中定义了模型的各个组件。
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, dropout=0.1):
        super(Transformer, self).__init__()
                         #用父类（即 nn.Module 类）的构造函数，以确保父类中定义的初始化逻辑得以执行。
        self.embed = nn.Embedding(vocab_size, embed_dim)
                         #embed是一个嵌入层，用于将输入的词索引映射为词嵌入向量
        self.positional_encoding = PositionalEncoding(embed_dim)
                         #positional_encoding 是位置编码层，用于为输入序列中的每个位置添加位置信息
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
                         #transformer_blocks 是一个由多个 TransformerBlock 组成的列表，用于构建整个Transformer的多层结构
        self.fc = nn.Linear(embed_dim, vocab_size)
                         #fc 是最后的线性层，用于将Transformer的输出映射为词汇表的大小。
            
            
            
    ##forward 方法中定义了模型的前向传播逻辑。首先，将输入数据通过嵌入层和位置编码层进行处理，然后依次通过多个Transformer块，最后经过线性层得到输出。
    def forward(self, x):
        x = self.embed(x)
        x = self.positional_encoding(x)#处理数据
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x) #通过多个Transformer块
        x = self.fc(x)   
        return x                     #线性层得到输出
                         
```

##### 什么是nn.Module 类

`nn.Module` 类是 PyTorch 中用于构建神经网络模型的基类。所有自定义的神经网络模型都应该继承自 `nn.Module` 类，并实现其核心方法，例如 `__init__` 和 `forward`。

`nn.Module` 提供了许多有用的功能和属性，使得神经网络的构建和训练更加方便和灵活。通过继承 `nn.Module` 类，可以使用其内置的函数和方法来管理模型中的参数、计算前向传播、初始化模型等。

在继承 `nn.Module` 类后，需要实现 `__init__` 方法来定义模型的结构和初始化操作。通常在 `__init__` 方法中定义模型的各个层和组件，并将它们作为类的属性。这些层和组件可以是卷积层、全连接层、池化层、循环层等，以构建所需的网络结构。

另外，还需要实现 `forward` 方法来定义模型的前向传播逻辑。在 `forward` 方法中，给定输入数据，应该描述数据在模型中的流动过程，即从输入到输出的计算过程。通过调用已定义的层和组件，并按照网络结构将数据传递给它们，可以实现前向传播的计算过程。

继承 `nn.Module` 类的好处是，它提供了自动追踪和管理模型的参数，可以方便地访问和修改模型的各个层和参数。此外，`nn.Module` 还提供了许多其他有用的功能，如模型的保存和加载、模型参数的初始化、自定义层的定义等。

通过继承 `nn.Module` 类并实现必要的方法，可以更好地组织和管理神经网络模型，使其更具可读性、可扩展性和可重用性。

#### 2.`PositionalEncoding`（位置编码类）

```py
class PositionalEncoding(nn.Module):
    ##__init__ 方法中定义了位置编码矩阵。通过生成正弦和余弦函数值，并根据位置和维度进行适当的填充，生成位置编码矩阵。位置编码是一个可学习的参数，将会与输入相加，用于为输入序列的每个位置提供位置信息。
    def __init__(self, embed_dim, max_seq_len=500):######修改最大句子长度
        
        super(PositionalEncoding, self).__init__()
                          #用父类（即 nn.Module 类）的构造函数，以确保父类中定义的初始化逻辑得以执行。nn.Module类是 PyTorch 中用于构建神经网络模型的基类。
            
        position = torch.arange(0, max_seq_len).unsqueeze(1)
                          #position是一个包含从 0 到 max_seq_len-1 的整数序列，通过调用 torch.arange() 和 unsqueeze() 函数得到。
                          		#torch.arange()是PyTorch 中的一个函数，用于创建一个张量，其中包含从起始值到结束值的连续整数序列。例如，torch.arange(0, 5) 会生成一个包含 0、1、2、3、4 的张量。
                          		#unsqueeze()是PyTorch 中的一个函数，用于在指定维度上给张量增加一个维度。例如，对于形状为 (3,) 的一维张量，调用 unsqueeze(0) 会将其形状变为 (1, 3)，在第一个维度上增加了一个维度。
                    
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))
                          #div_term 是一个计算位置编码中的分母项，它是一个张量，根据论文中的公式计算得到。
                          		#论文是 "Attention is All You Need"，这是指 Transformer 模型最初被提出的论文，由 Vaswani 等人在2017年发布。
                
        self.positional_encoding = nn.Parameter(torch.zeros(max_seq_len, embed_dim))
        self.positional_encoding.data[:, 0::2] = torch.sin(position * div_term)
        self.positional_encoding.data[:, 1::2] = torch.cos(position * div_term)
                          #这几行代码创建了一个可学习的参数 positional_encoding，它是一个形状为 (max_seq_len, embed_dim) 的零张量，并使用 torch.sin() 和 torch.cos() 函数对其进行位置和维度进行分半填充并初始化。通过使用 nn.Parameter，该参数将被自动注册为模型的一部分，并且在训练过程中可以被优化。
                           		#nn.Parameter见下：
                
                
                
    ##将输入数据与位置编码相加，以将位置编码应用到输入上。
    def forward(self, x):
        x = x + self.positional_encoding[:x.size(1), :]
        return x
                          #PositionalEncoding 类的前向传播逻辑。它接受一个输入张量 x，并将位置编码加到输入张量中。位置编码是通过取 positional_encoding 参数的子集，根据输入张量的长度进行切片操作得到的。最后，它返回已经添加了位置编码的张量 x。
```

##### 什么是nn.Parameter

`nn.Parameter` 是 `nn.Module` 的子类，用于定义模型中需要被训练的参数。它是一种特殊的张量，具有两个主要特点：

1. 自动注册：当将 `nn.Parameter` 对象作为模型的属性时，它会自动被注册为模型的可训练参数，并包含在模型的参数列表中。这意味着在模型的训练过程中，优化器将会更新这些参数的数值，以最小化损失函数。
2. 梯度计算：与一般的张量不同，`nn.Parameter` 对象会自动跟踪关于其自身的梯度。在反向传播过程中，通过调用损失函数的 `backward` 方法，可以计算并累积 `nn.Parameter` 对象的梯度，然后使用优化器进行参数更新。

在上述代码中，`nn.Parameter` 被用于定义 `PositionalEncoding` 类中的 `positional_encoding` 属性。这个属性用于存储位置编码的张量，并且它是一个可训练的参数。在模型训练过程中，优化器会根据损失函数的梯度更新 `positional_encoding` 的数值，以适应训练数据。

需要注意的是，`nn.Parameter` 对象通常在模型的构建过程中使用，并通过模型的 `parameters()` 方法来访问和管理。可以通过 `model.parameters()` 来获取模型中所有的 `nn.Parameter` 对象，并进行相应的操作，例如访问参数的值、修改参数的数值等。

#### 3.`TransformerBlock `（Transformer块类）

```py
class TransformerBlock(nn.Module):
    ##__init__ 方法中定义了Transformer块的各个组件。其中，attention 是多头注意力机制模块，dropout1 是第一个dropout层，norm1 是第一个层归一化层，fc 是前馈神经网络模块，dropout2 是第二个dropout层，norm2 是第二个层归一化层。
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout):
        super(TransformerBlock, self).__init__()

        self.attention = MultiheadAttention(embed_dim, num_heads)
                           #是多头注意力机制（Multihead Attention）模块，用于对输入进行自注意力计算。
            
        self.dropout1 = nn.Dropout(dropout)
                           #是第一个 Dropout 层，用于在训练过程中对 x 进行随机失活（dropout）操作。
            
                           		#随机失活（dropout）是深度学习中一种常用的正则化技术，旨在减少模型的过拟合。它通过在训练过程中以一定的概率随机丢弃（将其值设为零）神经网络中的部分神经元，从而减少神经元之间的依赖关系。
                
        self.norm1 = nn.LayerNorm(embed_dim)
                           #是第一个层归一化（Layer Normalization）层，用于对 residual + x 进行归一化操作。
            					#归一化（Layer Normalization）层见下：
                
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
                          #是前馈神经网络（Feedforward Neural Network）模块，它包含两个线性层和一个激活函数，用于对 x 进行非线性变换。
            
        self.dropout2 = nn.Dropout(dropout)
         				  #是第二个 Dropout 层，用于在训练过程中对 x 进行随机失活操作。
            
        self.norm2 = nn.LayerNorm(embed_dim)
                          #是第二个层归一化层，用于对 residual + x 进行归一化操作。
            
            
            
    ##forward 方法中定义了Transformer块的前向传播逻辑。首先，对输入进行残差连接，然后通过多头注意力机制，再经过dropout、层归一化，接着通过前馈神经网络，最后再次经过dropout、层归一化，并返回结果。
    def forward(self, x):
        residual = x
                          #将输入 x 保存在 residual 中，用于残差连接。
            
        x = self.attention(x)
                          #将 x 输入到多头注意力机制模块 self.attention 中进行计算，得到注意力输出。
        x = self.dropout1(x)
                          #对注意力输出进行第一个 Dropout 操作，使用 self.dropout1。
            
        x = self.norm1(residual + x)
                          #将残差连接和注意力输出相加，再通过第一个层归一化操作 self.norm1 进行归一化。 
        residual = x      #将输入 x 保存在 residual 中，用于残差连接。
        x = self.fc(x)    #将上述结果输入到前馈神经网络模块 self.fc 中进行变换。
        x = self.dropout2(x)
                          #对前馈神经网络的输出进行第二个 Dropout 操作，使用 self.dropout2。
        x = self.norm2(residual + x)
                          #再次将残差连接和前馈神经网络的输出相加，再通过第二个层归一化操作 self.norm2 进行归一化。
            
        return x          #将得到的结果作为 Transformer 块的输出。
    
    ##这样，每个 Transformer 块都会对输入进行自注意力计算、非线性变换，并保留了残差连接和层归一化操作，以帮助模型更好地捕捉序列中的信息和建模能力。
```

##### 什么是归一化

归一化（Normalization）层是深度学习模型中常用的一种技术，用于在网络的不同层之间对输入数据进行归一化处理。它的目的是使输入数据在经过归一化后具有相似的分布特性，有助于提高模型的训练效果和泛化能力。

归一化层的一种常见类型是层归一化（Layer Normalization），它是在每个样本的特征维度上进行归一化操作。与批归一化（Batch Normalization）不同，批归一化是在每个批次的样本上进行归一化操作。

在 Transformer 模型中，层归一化被广泛应用于每个 Transformer 块中的不同子层（如自注意力层、前馈神经网络层等）。它的作用是对每个样本的特征维度进行归一化，使得每个维度上的特征具有相似的分布。这有助于减少模型对输入数据中的特定维度的依赖，并提高模型的鲁棒性和学习效果。

具体而言，层归一化层的计算过程如下：

1. 对于输入张量 `x`，计算每个样本在特征维度上的均值和标准差，得到 `mean` 和 `std`。
2. 对于每个样本，将输入张量 `x` 减去均值 `mean`，然后除以标准差 `std`，得到归一化后的输出。
3. 对归一化后的输出进行缩放和平移操作，使用可学习的参数 `gamma` 和 `beta`，以便模型可以根据需要调整数据的范围和偏移。

层归一化的引入有助于解决深度神经网络中的梯度消失和梯度爆炸问题，使得网络的训练更加稳定和高效。此外，它还可以提供一定程度的正则化效果，有助于减少模型的过拟合。

总之，层归一化是一种常用的归一化技术，在 Transformer 模型中用于对输入数据的特征维度进行归一化处理，以提高模型的学习效果和鲁棒性。

#### 4.`MultiheadAttention` (多头注意力机制类)

```py
class MultiheadAttention(nn.Module):
    ##定义了多头注意力机制的各个组件。其中，q_linear、k_linear 和 v_linear 是三个线性层，用于将输入数据映射到查询（q）、键（k）和值（v）的空间。fc 是最后的线性层，用于将多头注意力机制的输出映射回原始维度。这是 Transformer 模型中关键的组件之一，用于捕捉输入序列的全局关联信息。
    def __init__(self, embed_dim, num_heads):
        super(MultiheadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  #导入数据

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)
                          #定义了几个线性层 (q_linear, k_linear, v_linear, fc)，用于将输入数据映射到查询、键和值的空间，以及将多头注意力机制的输出映射回原始维度。 
            
    ##定义了多头注意力机制的前向传播逻辑。首先，通过线性层将输入数据映射到查询、键和值的空间，并对其进行形状变换。然后，计算注意力分数、注意力权重，得到加权的值表示。最后，通过线性层将加权值映射回原始维度，并返回结果。
    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                          #在方法的实现中，首先获取输入张量的维度信息，包括批量大小 (batch_size)、序列长度 (seq_len) 和输入维度 (embed_dim)。然后，通过线性层将输入数据映射到查询 (q)、键 (k) 和值 (v) 的空间，并对它们进行形状变换，使得每个头的维度处于正确的位置。
            

        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
                          #通过将查询和键进行点积运算，并除以 sqrt(self.head_dim) 进行缩放。
        attention_weights = torch.softmax(scores, dim=-1)
                          #应用 softmax 激活函数，计算注意力权重 (attention_weights)。

        x = torch.matmul(attention_weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
                          #将注意力权重与值进行加权求和，得到加权的值表示。然后，进行形状变换和转置操作，使得头的维度位于正确的位置，并通过 contiguous() 方法确保张量的内存是连续的。
        x = self.fc(x)    #通过线性层 fc 将加权值映射回原始维度，并返回结果。
        return x
```











```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
import jieba

class ChineseTextDataset(Dataset):
    def __init__(self, file_path):
        self.sentences = []
        self.inputs = []
        self.targets = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    sentence = line.split("[MASK]")
                    self.sentences.append(line)
                    input_tokens = list(jieba.cut(sentence[0].strip()))
                    target_tokens = list(jieba.cut(sentence[1].strip()))
                    self.inputs.append(input_tokens)
                    self.targets.append(target_tokens)
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

class ChineseTransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dropout=0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout,
        )
        self.predictor = nn.Linear(d_model, vocab_size)
        
    def forward(self, inputs, targets):
        input_embed = self.embedding(inputs)
        target_embed = self.embedding(targets)
        input_embed = input_embed.permute(1, 0, 2)
        target_embed = target_embed.permute(1, 0, 2)
        mask = self.transformer.generate_square_subsequent_mask(targets.size(0)).to(inputs.device)
        output = self.transformer(input_embed, target_embed, mask)
        output = output.permute(1, 0, 2)
        output = self.predictor(output)
        return output

# 超参数
vocab_size = 5000  # 自定义词汇表大小
d_model = 256
nhead = 4
num_layers = 4
dropout = 0.1
lr = 1e-4
num_epochs = 10
batch_size = 16

# 初始化模型
model = ChineseTransformerModel(vocab_size, d_model, nhead, num_layers, dropout)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 初始化数据加载器
file_path = "data.txt"  # 数据集文件路径
dataset = ChineseTextDataset(file_path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# 训练模型
for epoch in range(num_epochs):
    total_loss = 0.0
    for inputs, targets in dataloader:
        input_tokens = [torch.tensor([vocab_size if token not in input_tokens else input_tokens.index(token) for token in input_tokens]) for input_tokens in inputs]
        target_tokens = [torch.tensor([vocab_size if token not in target_tokens else target_tokens.index(token) for token

 in target_tokens]) for target_tokens in targets]
        inputs = nn.utils.rnn.pad_sequence(input_tokens, batch_first=True).to(device)
        targets = nn.utils.rnn.pad_sequence(target_tokens, batch_first=True).to(device)

        optimizer.zero_grad()

        output = model(inputs, targets)
        loss = criterion(output.view(-1, vocab_size), targets.view(-1))
        loss.backward()

        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")

# 预测
input_sentence = "[BEGIN]事实证明 8M 参[MASK]就能做一个差强人意的模型出来。[END]"
input_tokens = list(jieba.cut(input_sentence.strip()))
input_tokens = [torch.tensor([vocab_size if token not in input_tokens else input_tokens.index(token) for token in input_tokens])]

output_tokens = model(input_tokens, input_tokens)
output_tokens = output_tokens.argmax(dim=-1).squeeze().tolist()

output_sentence = ''.join([input_tokens[0][index] if index < len(input_tokens[0]) else "" for index in output_tokens])
print(output_sentence)
```

请确保已安装了 jieba 库，并在代码中引入了它。我已经更新了代码，以使用 jieba 库作为分词器，并在预测阶段生成输出句子。

在预测阶段，将输入句子进行分词，并将每个词转换为索引，然后传递给模型进行预测。最后，根据模型的输出索引，还原为实际的中文句子。

### 二.数据预加载与模型保存 

#### 导入数据集

```python
# 定义数据集
dataset = [
    "[BEGIN]事实证明 8M 参[MASK]就能做[MASK]差强人意的模型出来。[END]",
    "[BEGIN]这是一个例子，你可以[MASK]据需要添加更多的句子。[END]"
]

with open('234.txt', 'r', encoding='utf-8') as f:#与训练时一致
    lines = f.readlines()
    for line in lines:
        dataset.append(line.strip())
print('finish!')
```

#### 构建数据表

```py 
# 构建词汇表，将输入文本和目标文本（这里是相同的文本）转换为对应的索引序列，并将它们存储在 input_seqs 和 target_seqs 列表中。这是为了在后续的模型训练和处理中能够使用索引表示的数据。
vocab = set()
for sentence in dataset:
    vocab.update(list(sentence))
                          #遍历数据集中的每个句子，将句子中的字符添加到词汇表集合中。
vocab = list(vocab)       #将词汇表集合转换为列表，以便索引操作和排序。
vocab_size = len(vocab)   #词汇表大小
print("词汇表大小:")        
print(vocab_size)         #输出vocab_size的大小--------用来在重新训练时输入对应的size避免张量报错
char2idx = {c: i for i, c in enumerate(vocab)}
                          #创建一个字典 char2idx，将词汇表中的字符映射到对应的索引值。
idx2char = {i: c for i, c in enumerate(vocab)}
                          #创建一个字典 idx2char，将索引值映射回词汇表中的字符。

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
                         #遍历数据集中的每个句子，将句子中的[MASK]标记替换为空字符串，得到输入文本。
                         #标记替换为空字符串，得到目标文本，这里即与输入文本相同。
    input_seq = [char2idx.get(c, char2idx[unknown_token]) for c in input_text]
    target_seq = [char2idx.get(c, char2idx[unknown_token]) for c in target_text]
                         #将目标文本中的每个字符转换为对应的索引值，并存储在 target_seq 列表中。如果字符不在词汇表中，则将其映射为未知字符的索引。
    input_seqs.append(input_seq)
    target_seqs.append(target_seq)
                         #将输入和目标序列添加到相应的列表中。
```

#### 预加载与保存模型

```py
# 创建并初始化模型
model = Transformer(vocab_size, embed_dim, num_heads, hidden_dim, num_layers).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001) 
                                #创建一个 Adam 优化器，用于优化模型的参数。学习率设置为 0.001。
criterion = nn.CrossEntropyLoss()
                                #创建一个交叉熵损失函数，用于计算模型输出和目标之间的损失。

# 将数据转换为Tensor即 PyTorch 的张量，并将它们移动到指定的设备上。
input_tensors = [torch.tensor(seq).to(device) for seq in input_seqs]
target_tensors = [torch.tensor(seq).to(device) for seq in target_seqs]

# 训练模型
num_epochs = 1                  #训练轮数
batch_size = 256                #批次大小
total_batches = len(input_tensors)

for epoch in range(num_epochs):
    model.train()               #将模型设置为训练模式，启用 dropout 和批归一化层的训练行为
    total_loss = 0
    for step, (input_tensor, target_tensor) in enumerate(zip(input_tensors, target_tensors), 1):
        optimizer.zero_grad()   #清除优化器中之前的梯度信息
        output_tensor = model(input_tensor.unsqueeze(0))
                                #将输入张量传递给模型进行前向传播，得到输出张量。
        loss = criterion(output_tensor.squeeze(0)[:-1, :], target_tensor[:-1])
                                #计算模型输出与目标之间的损失。由于 Transformer 模型通常会输出一个比目标序列长度多一个时间步的序列，所以在计算损失时需要将输出和目标序列的长度对齐
        loss.backward()         #反向传播计算梯度
        optimizer.step()        #更新模型参数，执行优化器的一步参数更新。 
        total_loss += loss.item()
                                #累加当前步骤的损失值到总损失中

        # 打印每个步骤的进度
        print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{step}/{total_batches}], Loss: {loss.item():.4f}")

    avg_loss = total_loss / total_batches
    print(f"Avg Loss: {avg_loss:.4f}")

# 保存模型
torch.save(model.state_dict(), "model.pth")
```



#### 试着进行预测

```py
# 使用模型进行预测
input_text = "[BEGIN]事实证明 8M 参[MASK]就能做[MASK]差强人意的模型出来。[END]"

input_seq = [char2idx.get(c, char2idx[unknown_token]) for c in input_text]
							#将输入文本转换为序列形式，将每个字符映射为对应的索引值。如果字符不在字符到索引的映射字典char2idx中，就使用未知标记unknown_token对应的索引值。生成的输入序列存储在input_seq中。
input_tensor = torch.tensor(input_seq).unsqueeze(0).to(device)
                            #将输入序列转换为PyTorch张量，并在第0维上添加一个维度，然后将其发送到设备（如GPU）上进行计算。这里使用的设备是之前定义的device。
output_tensor = model(input_tensor)
							#将输入张量传递给模型进行预测，得到输出张量output_tensor。模型会根据输入序列中的[MASK]标记来预测相应位置的字符。
output_seq = torch.argmax(output_tensor.squeeze(0), dim=-1)
							#对输出张量进行处理，通过取最大值的索引来获取预测的字符索引序列output_seq。使用torch.argmax函数在维度dim=-1上进行操作，这将返回每个位置上概率最大的字符索引。

output_text = input_text
mask_count = 0
for i in range(len(output_text)):
    if output_text[i:i + len('[MASK]')] == '[MASK]':
        					#使用循环遍历输出文本中的每个字符，检查是否出现[MASK]标记。如果存在[MASK]标记，则执行以下操作
        mask_count += 1
        if mask_count <= len(output_seq):
            				#检查mask_count是否小于等于预测序列的长度，以确保仍有可用的预测字符
            predicted_char = idx2char[output_seq[mask_count - 1].item()]
            				#通过索引output_seq[mask_count - 1].item()从索引到字符的映射字典idx2char中获取实际字符。
            while re.match(r'[a-zA-Z0-9\[\]\s]', predicted_char) or predicted_char in ['[MASK]', '[BEGIN]', '[END]'] or predicted_char == '':
                			#使用正则表达式和条件检查，确保预测字符不是字母、数字、空格、[MASK]、[BEGIN]、[END]之一，也不是空字符。如果不满足条件，则进行以下操作：
                new_output_seq = torch.argmax(output_tensor.squeeze(0), dim=-1)
                			#创建一个新的输出序列new_output_seq，并将其设置为输出张量的副本。
                new_output_seq[mask_count - 1] = torch.randint(low=0, high=vocab_size, size=(1,)).item()           #将第mask_count - 1个位置的索引设置为随机整数，以表示一个随机字符。
                predicted_char = idx2char[new_output_seq[mask_count - 1].item()]
            output_text = output_text[:i] + predicted_char + output_text[i + len('[MASK]'):]
        else:
            output_text = output_text[:i] + '' + output_text[i + len('[MASK]'):]
            				#将预测字符插入到输出文本中，替换原来的[MASK]标记。
print("Input Text:", input_text)
print("Output Text:", output_text)
```

以下是小小的结果（无法复现）：

```
input_text = "[BEGIN]事实证明 8M 参[MASK]就能做[MASK]差强人意的模型出来。[END]"
output_text = "[BEGIN]事实证明 8M 参宿就能做很差强人意的模型出来。[END]"
correct_text = "[BEGIN]事实证明 8M 参数就能做出差强人意的模型出来。[END]"
```



### 三.数据重加载与模型保存与使用与反馈训练

#### 重新导入必要的库 

```py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import re


# 检查是否有可用的GPU设备
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# 定义模型参数
embed_dim = 64      # 词向量维度/2
num_heads = 2       #多头数
hidden_dim = 32    # 隐藏层维度/4
num_layers = 4      # 网络层数
max_seq_len = 600  #最大句子长度（仅做标记）
#在retrain中/1.4才是输入最大列

# 设置随机种子以便复现性
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

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
vocab_size = 5798###################################################################################len(vocab)   #词汇表大小与上文一致，避免报错
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

# 定义模型和其他组件（同上，这里仅仅展示Transformer类）
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
#-------------------------------------------------------------
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

# 使用模型进行预测并保存于output_texts
input_texts = ["[BEGIN]事实证明 8M 参[MASK]就能做[MASK]差强人意的模型出来。[END]", "[BEGIN]这是另一[MASK]例子，你可以根据需要添加更多的句子。[END]"]
with open('234.txt', 'r', encoding='utf-8') as f:#与训练时一致
    lines = f.readlines()
    for line in lines:
        input_texts.append(line.strip())
print('finish!')

correct_count = 0
total_count = 0



# 创建并初始化模型
model = Transformer(vocab_size, embed_dim, num_heads, hidden_dim, num_layers).to(device)
model.load_state_dict(torch.load("model.pth"))
model.eval()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 将数据转换为Tensor
input_tensors = [torch.tensor(seq).to(device) for seq in input_seqs]
target_tensors = [torch.tensor(seq).to(device) for seq in target_seqs]

# 训练模型(反馈训练，见下)
num_epochs = 1
batch_size = 256
train_model(model, optimizer, criterion, input_tensors, target_tensors, correct_texts, num_epochs, batch_size)

# 保存模型
torch.save(model.state_dict(), "model.pth")


```

#### 反馈训练

```py
#函数`train_model`接受模型 (`model`)、优化器 (`optimizer`)、损失函数 (`criterion`)、输入张量 (`input_tensors`)、目标张量 (`target_tensors`)、正确文本 (`correct_texts`)、训练周期数 (`num_epochs`) 和批次大小 (`batch_size`) 作为参数。

# 反馈训练
def train_model(model, optimizer, criterion, input_tensors, target_tensors, correct_texts, num_epochs, batch_size):
    total_batches = len(input_tensors)
                          #将模型设置为训练模式，通过 `model.train()` 实现。
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
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
            				#将输入文本转换为序列形式，将每个字符映射为对应的索引值，与之前的代码段相同。
            input_tensor = torch.tensor(input_seq).unsqueeze(0).to(device)
            				#将输入序列转换为PyTorch张量，并在第0维上添加一个维度，然后将其发送到设备上进行计算，与之前的代码段相同。
            output_tensor = model(input_tensor)
            				#初始化输出文本 `output_text` 为与输入文本相同的初始值。
            output_seq = torch.argmax(output_tensor.squeeze(0), dim=-1)
            				
            output_text = input_text
            mask_count = 0
            for i in range(len(output_text)):
                if output_text[i:i + len('[MASK]')] == '[MASK]':
                    		#对输出文本中的每个字符进行遍历，检查是否出现 `[MASK]` 标记，与之前的代码段相同。
                    mask_count += 1
                    if mask_count <= len(output_seq):
                        predicted_char = idx2char[output_seq[mask_count - 1].item()]
                        while re.match(r'[a-zA-Z0-9\[\]\s]', predicted_char) or predicted_char in ['[MASK]', '[BEGIN]', '[END]'] or predicted_char == '':
                            new_output_seq = torch.argmax(output_tensor.squeeze(0), dim=-1)
                            new_output_seq[mask_count - 1] = torch.randint(low=0, high=vocab_size, size=(1,)).item()
                            predicted_char = idx2char[new_output_seq[mask_count - 1].item()]
                           
                        output_text = output_text[:i] + predicted_char + output_text[i + len('[MASK]'):]			#将预测字符插入到输出文本中，替换原来的 `[MASK]` 标记，与之前的代码段相同。
                    else:
                        output_text = output_text[:i] + '' + output_text[i + len('[MASK]'):]
            print("Input Text:", input_text)
            print("Output Text:", output_text)

            correct_text=correct_texts[weizhi]
            				#在每个批次结束后，通过位置 `weizhi` 更新正确文本的位置。
            print ("Correct Text",correct_text)
            weizhi = weizhi + 1             #位置更新

            # 将正确文本与生成的输出文本转换为Tensor
            correct_tensor = torch.tensor([char2idx.get(c, char2idx[unknown_token]) for c in correct_text]).unsqueeze(0).to(device)
            output_tensor = torch.tensor([char2idx.get(c, char2idx[unknown_token]) for c in output_text]).unsqueeze(0).to(device)
            								# #将正确文本和生成的输出文本转换为张量形式，与之前的代码段相同。
            output_tensor = output_tensor.float()
            correct_tensor = correct_tensor.float()               #加入float类型
            output_tensor = output_tensor.requires_grad_()
            correct_tensor = correct_tensor.requires_grad_()      #加入grad要求避免报错为了进行反向传播和计算梯度。

            # 计算损失并进行反向传播和优化
            loss = criterion(output_tensor.squeeze(0)[:-1], correct_tensor.squeeze(0)[:-1])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 				#清空优化器，反向传播并执行优化器的一步更新，通过 `optimizer.step()` 实现。

            print("Loss:", loss.item()/100000000)
            print("-----------------------------------------")
```



------------------定义一个`transformer`(模型类)里面有

​      一个嵌入层，用于将输入的词索引映射为词嵌入向量
​      位置编码层，用于为输入序列中的每个位置添加位置信息
​       多个 TransformerBlock 组成的列表，用于构建整个Transformer的多层结构
​        最后的线性层，用于将Transformer的输出映射为词汇表的大小。

​        将输入的x依次嵌入层和位置编码层进行处理，然后依次通过多个Transformer块，最后经过线性层得到输出并前向传播。

------------------定义一个PositionalEncoding（位置编码类）

通过生成正弦和余弦函数值，并根据位置和维度进行适当的填充，可以确保位置编码矩阵具有一定的周期性和变化性，从而提供丰富的位置信息。不同的位置会得到不同的编码，而且相对位置之间的关系也能得到一定程度的捕捉。这样，模型在处理序列时可以更好地感知不同位置之间的距离和顺序，从而更好地进行推理和建模。

```py

                
        self.positional_encoding = nn.Parameter(torch.zeros(max_seq_len, embed_dim))
        self.positional_encoding.data[:, 0::2] = torch.sin(position * div_term)
        self.positional_encoding.data[:, 1::2] = torch.cos(position * div_term)
                          #这几行代码创建了一个可学习的参数 positional_encoding，它是一个形状为 (max_seq_len, embed_dim) 的零张量，并使用 torch.sin() 和 torch.cos() 函数对其进行位置和维度进行分半填充并初始化。通过使用 nn.Parameter，该参数将被自动注册为模型的一部分，并且在训练过程中可以被优化。           
                
```

`TransformerBlock `（Transformer块类）

---------------第一个 Dropout 层，用于在训练过程中对 x 进行随机失活（dropout）操作。
            

    #随机失活（dropout）是深度学习中一种常用的正则化技术，旨在减少模型的过拟合。它通过在训练过程中以一定的概率随机丢弃（将其值设为零）神经网络中的部分神经元，从而减少神经元之间的依赖关系。

```py
class TransformerBlock(nn.Module):
    ##__init__ 方法中定义了Transformer块的各个组件。其中，attention 是多头注意力机制模块，dropout1 是第一个dropout层，norm1 是第一个层归一化层，fc 是前馈神经网络模块，dropout2 是第二个dropout层，norm2 是第二个层归一化层。
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout):
        super(TransformerBlock, self).__init__()

        self.attention = MultiheadAttention(embed_dim, num_heads)
                           #是多头注意力机制（Multihead Attention）模块，用于对输入进行自注意力计算。
            
        self.dropout1 = nn.Dropout(dropout)
                           #是
                
        self.norm1 = nn.LayerNorm(embed_dim)
                           #是第一个层归一化（Layer Normalization）层，用于对 residual + x 进行归一化操作。
            					#归一化（Layer Normalization）层见下：
 
    ##forward 方法中定义了Transformer块的前向传播逻辑。首先，对输入进行残差连接，然后通过多头注意力机制，再经过dropout、层归一化，接着通过前馈神经网络，最后再次经过dropout、层归一化，并返回结果。
```



具体而言，层归一化层的计算过程如下：

1. 对于输入张量 `x`，计算每个样本在特征维度上的均值和标准差，得到 `mean` 和 `std`。
2. 对于每个样本，将输入张量 `x` 减去均值 `mean`，然后除以标准差 `std`，得到归一化后的输出。
3. 对归一化后的输出进行缩放和平移操作，使用可学习的参数 `gamma` 和 `beta`，以便模型可以根据需要调整数据的范围和偏移。



`MultiheadAttention` (多头注意力机制类)

```py
class MultiheadAttention(nn.Module):
    ##定义了多头注意力机制的各个组件。其中，q_linear、k_linear 和 v_linear 是三个线性层，用于将输入数据映射到查询（q）、键（k）和值（v）的空间。fc 是最后的线性层，用于将多头注意力机制的输出映射回原始维度。这是 Transformer 模型中关键的组件之一，用于捕捉输入序列的全局关联信息。
    def __init__(self, embed_dim, num_heads):
        super(MultiheadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  #导入数据

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)
                          #定义了几个线性层 (q_linear, k_linear, v_linear, fc)，用于将输入数据映射到查询、键和值的空间，以及将多头注意力机制的输出映射回原始维度。 
            
    ##定义了多头注意力机制的前向传播逻辑。首先，通过线性层将输入数据映射到查询、键和值的空间，并对其进行形状变换。然后，计算注意力分数、注意力权重，得到加权的值表示。最后，通过线性层将加权值映射回原始维度，并返回结果。
    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                          #在方法的实现中，首先获取输入张量的维度信息，包括批量大小 (batch_size)、序列长度 (seq_len) 和输入维度 (embed_dim)。然后，通过线性层将输入数据映射到查询 (q)、键 (k) 和值 (v) 的空间，并对它们进行形状变换，使得每个头的维度处于正确的位置。
            

        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
                          #通过将查询和键进行点积运算，并除以 sqrt(self.head_dim) 进行缩放。
        attention_weights = torch.softmax(scores, dim=-1)
                          #应用 softmax 激活函数，计算注意力权重 (attention_weights)。

        x = torch.matmul(attention_weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
                          #将注意力权重与值进行加权求和，得到加权的值表示。然后，进行形状变换和转置操作，使得头的维度位于正确的位置，并通过 contiguous() 方法确保张量的内存是连续的。
        x = self.fc(x)    #通过线性层 fc 将加权值映射回原始维度，并返回结果。
        return x
```



### 四.压缩模型

我希望能够将训练后的模型进行剪枝操作减小它的大小，如下：

```py
#定义剪枝模型
def prune_model(model, prune_percent):
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=prune_percent,
    )

    return model
```

在压缩后的模型大小大约46M



### 五.网络可视化

我在使用tensorwatch中遇到了版本不兼容的问题，而使用低等级的cuda版本会导致我的代码报错（o(╥﹏╥)o）

下面是一个示例代码，将训练过程中的损失值可视化，并在训练结束后保存模型，并可视化模型的网络结构。

```python
import tensorwatch as tw
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 创建观察者对象
watcher = tw.Watcher()

# 监视损失值
loss_tracker = watcher.create_stream(name='Loss', display_name='Loss')

# 定义训练函数
def train(model, dataloader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # 计算平均损失值
        average_loss = running_loss / len(dataloader)

        # 更新观察者对象中的损失值
        watcher.update(loss_tracker, average_loss)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {average_loss:.4f}")

# 训练模型
train(model, dataloader, criterion, optimizer, epochs=10)

# 可视化损失值的变化
loss_histogram = tw.Histogram(loss_tracker.get_data(), bins=10)
loss_histogram.plot(title='Loss Histogram')

# 保存模型
torch.save(model.state_dict(), 'model.pth')

# 可视化模型的网络结构
model_graph = tw.draw_model(model, [1, 3, 32, 32])
model_graph.save('model_graph.png')
```

在这个示例中，我首先创建了一个 `watcher` 对象，并使用 `watcher.create_stream` 方法创建了一个用于跟踪损失值的观察者流 (`loss_tracker`)。然后，在训练循环中，我们在每个批次结束时更新了观察者对象中的损失值。最后，我们使用 `tw.Histogram` 类创建了损失值的直方图，并调用 `plot` 方法进行可视化。

同时，我们还使用了 `tw.draw_model` 方法可视化了模型的网络结构，并将其保存为 `model_graph.png` 文件。



