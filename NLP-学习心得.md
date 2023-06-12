## PythonAI第四轮，NLP学习心得

需求分析：

1.基于 Attention 机制搭建一个 Transform 模型，模型可以参考 pytorch 的教程、参考开源

模型如 Bert、GPT 等

2.训练时采用无监督学习，预测掩蔽字的方法训练，具体方法例如：

3.整个模型不能超过 50M（五千万）



### 如何构建掩蔽字函数



```py 
def predict_next_token(model, device, inputs, last_output, temperature=1.0):
    logits = last_output[-1, :]
    logits = logits / temperature
    probabilities = F.softmax(logits, dim=-1)
    predicted_token = torch.multinomial(probabilities, num_samples=1).squeeze()
    return predicted_token.item()

# 生成句子
num_sentences = 5
max_length = 20
start_token = '<start>'
end_token = '<end>'
mask_token = '<mask>'
temperature = 1.0 #尝试调整温度参数来控制生成句子的多样性。较高的温度值会导致更随机的生成，而较低的温度值会使生成更加确定性。
```







### 定义模型



```py 
# 定义模型
class TransformerModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):
        super(TransformerModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(len(vocab2id), embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8, dim_feedforward=hidden_dim, dropout=dropout, activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.decoder = nn.Linear(embedding_dim, len(vocab2id))

    def forward(self, src):
        # src shape: [seq_len, batch_size]
        embedded = self.embedding(src) * np.sqrt(self.hidden_dim)
        embedded = self.pos_encoder(embedded)
        output = self.transformer_encoder(embedded)
        output = self.decoder(output)
        return output
```





```py
# 定义Positional Encoding层
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
```





### 句子生成



```py
def generate_pre_sentence(model, device, inputs, temperature, max_length=20):
    model.eval()
    sentence = []

    with torch.no_grad():
        while True:
            output = model(inputs)
            token = output.argmax(dim=-1)[-1].item()

            if token == vocab2id[end_token] or len(sentence) >= max_length:
                break

            token_str = id2vocab[token]

            if token_str == unk_token or token_str not in vocab2id:
                break

            sentence.append(token_str)

            if token_str == padding_token:
                continue

            token_tensor = torch.tensor([[vocab2id[token_str]]]).to(device)
            inputs = torch.cat([inputs, token_tensor], dim=1)

            if token_str == mask_token:
                token_str = generate_sentence(model, device, vocab2id, max_length, start_token, end_token, mask_token, temperature)[-1]
                token_tensor = torch.tensor([[vocab2id[token_str]]]).to(device)
                inputs = torch.cat([inputs, token_tensor], dim=1)

    return sentence



def generate_masked_sentence(model, device, inputs, mask_token, temperature, max_length=20):
    model.eval()
    sentence = []

    with torch.no_grad():
        while True:
            output = model(inputs)
            token = output.argmax(dim=-1)[-1].item()

            if token == vocab2id[end_token] or len(sentence) >= max_length:
                break

            token_str = id2vocab[token]

            if token_str == unk_token or token_str not in vocab2id:
                break

            sentence.append(token_str)

            if token_str == padding_token:
                continue

            if token_str == mask_token:
                token_str = generate_sentence(model, device, inputs, temperature)[-1]

            token_tensor = torch.tensor([[vocab2id[token_str]]]).to(device)
            inputs = torch.cat([inputs, token_tensor], dim=1)

    return sentence

# 生成未标记的句子示例
sentence = generate_sentence(model, device, inputs, temperature)

# 生成带有<mask>标记的句子示例
sentence_masked = generate_masked_sentence(model, device, inputs, mask_token, temperature)


def generate_sentence(model, device, vocab2id, max_length, start_token='<start>', end_token='<end>', mask_token='<mask>', temperature=1.0):
    inputs = torch.tensor([[vocab2id[start_token]]]).to(device)
    output = model(inputs)

    sentence = []
    while True:
        token = output.argmax(dim=-1)[-1].item()

        if token == vocab2id[end_token] or len(sentence) >= max_length:
            break

        token_str = id2vocab[token]

        if token_str == unk_token or token_str not in vocab2id:
            break

        sentence.append(token_str)

        if token_str == padding_token:
            continue

        if token_str == mask_token:
            token_str = generate_pre_sentence(model, device, inputs, temperature)[-1]

        token_tensor = torch.tensor([[vocab2id[token_str]]]).to(device)
        output = model(token_tensor)
        inputs = torch.cat([inputs, token_tensor], dim=1)  # 更新输入序列
    return ' '.join(sentence)
```







### 压缩模型



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





### 训练模型



```py
#定义训练函数
def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0
    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        # 计算损失并反向传播
        loss = criterion(outputs.view(-1, outputs.shape[-1]), targets.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 打印训练日志
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, N_EPOCHS, i + 1, len(train_loader),
                                                                     total_loss / (i + 1)))

    return total_loss / len(train_loader)
```



```py
# 加载训练数据
train_data = []
with open('C:\\Users\\Lenovo\\Desktop\\py\\NLP\\summary.txt', 'r', encoding='utf-8') as file:
    print('model start')
    # 逐行读取文件内容
    for line in file:
        # 去除行尾的换行符
        line = line.strip()
        # 将每行文本转换为以空格分隔的单词形式
        words = jieba.lcut(line)
        # 将单词列表转换为一个字符串，并添加到训练数据列表中
        train_data.append(' '.join(words))

    print('model end')
```

正常训练模型

```py
# 训练模型
for epoch in range(N_EPOCHS):
    print("start:")
    model.train()
    total_loss = 0

    for i, inputs in enumerate(train_loader):
        inputs = inputs.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        # 构建掩蔽字
        mask = (inputs == vocab2id['<mask>'])
        targets = inputs.masked_fill(mask, -100)  # 将掩蔽字的目标值设为-100

        # 计算损失并反向传播
        loss = criterion(outputs.view(-1, outputs.shape[-1]), targets.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 打印训练日志
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, N_EPOCHS, i+1, len(train_loader), total_loss / (i+1)))

# 保存模型
torch.save(model.state_dict(), 'transformer_model.pth')
print("模型已保存")

```





显存严重不足力：

```py
# 训练模型
#######################################################################################################################
# 将训练数据划分为多个小数据集
num_batches = 100
batch_size = len(train_data) // num_batches

# 分批训练和保存模型
for batch_idx in range(num_batches):
    # 获取当前批次的训练数据
    start_idx = batch_idx * batch_size
    end_idx = (batch_idx + 1) * batch_size
    train_data_subset = train_data[start_idx:end_idx]
    print(batch_idx)

    # 构建词汇表
    vocab2id, id2vocab = build_vocab(train_data_subset)
    vocab_size = len(vocab2id)
    # 补充‘<unk>’
    vocab2id['<unk>'] = len(vocab2id)
    id2vocab[len(vocab2id)] = '<unk>'
    # 补充‘<pad>’
    vocab2id['<pad>'] = len(vocab2id)
    id2vocab[len(vocab2id)] = '<pad>'
    vocab2id['<mask>'] = len(vocab2id)
    id2vocab[len(vocab2id)] = '<mask>'
    vocab2id['<start>'] = len(vocab2id)
    id2vocab[len(vocab2id)] = '<start>'
    vocab2id['<end>'] = len(vocab2id)
    id2vocab[len(vocab2id)] = '<end>'


    # 预处理数据
    train_data_subset = [preprocess(text, vocab2id) for text in train_data_subset]

    # 将数据转换为PyTorch张量
    train_data_subset = [torch.tensor(text) for text in train_data_subset]
    # 创建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerModel(EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT).to(device)

    # 填充数据
    train_data_subset = pad_sequence(train_data_subset, batch_first=True, padding_value=vocab2id['<pad>'])
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(train_data_subset, batch_size=BATCH_SIZE, shuffle=True)

    # 创建模型
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 梯度累积相关变量,减小显存压力
    accumulation_steps = 4  # 设置梯度累积的步数
    total_loss = 0

    # 训练模型
    for epoch in range(N_EPOCHS):
        print("start:")
        model.train()

        for i, inputs in enumerate(train_loader):
            inputs = inputs.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # 构建掩蔽字
            mask = (inputs == vocab2id['<mask>'])
            targets = inputs.masked_fill(mask, -100)  # 将掩蔽字的目标值设为-100

            # 计算损失
            loss = criterion(outputs.view(-1, outputs.shape[-1]), targets.view(-1))
            loss /= accumulation_steps  # 平均每个小批次的损失

            # 反向传播
            loss.backward()

            # 累积梯度
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()

            # 打印训练日志
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, N_EPOCHS, i + 1, len(train_loader),
                                                                     total_loss / (i + 1)))

        # 释放不需要的变量
        del train_data_subset, train_loader, inputs, outputs, targets, loss
        torch.cuda.empty_cache()  # 清空显存缓存
    # 在训练之后剪枝模型，例如剪枝50%的参数
    #model = prune_model(model, prune_percent=0.5)
    # 保存模型
    torch.save(model.state_dict(), f"model_batch{batch_idx}.pth")

#######################################################################################################################
```

