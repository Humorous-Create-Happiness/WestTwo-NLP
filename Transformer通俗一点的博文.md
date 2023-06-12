## Transformer通俗一点的博文

**论文：**[Attention is all you need](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1706.03762)

**来源：**[https://jalammar.github.io/illustrated-transformer/](https://link.zhihu.com/?target=https%3A//jalammar.github.io/illustrated-transformer/)

编者注：本文是对Jay Alammar的The Illustrated Transformer的中文翻译，由于直接翻译会产生误解，因此本文中会加入一些个人的理解，如有不妥，请留言讨论！

**正文：**

在之前的[博客](https://link.zhihu.com/?target=https%3A//jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)中，我们研究了Attention——一个在现代深度学习模型中无处不在的方法。Attention是一种有助于提高神经机器翻译的方法。在这个博客中，我们将重点关注The Transformer——一种利用Attention来加速模型训练的方法。The Transformer在一些特殊任务上超越了Google Neural Machine Translation model（RNN+Attention）。The Transformer最大的优势来源于它的并行化计算。实际上，Google Cloud建议使用Transformer作为参考模型来使用其[Cloud TPU](https://link.zhihu.com/?target=https%3A//cloud.google.com/tpu/)产品。所以，让我们来分解这个模型，看看它是如何工作的。

## **A High-Level Look（一个宏观的概括）**

首先，我们将Transformer看成一个简单的黑盒模型。在机器翻译任务下，将源语言（法语）的一个句子A输入其中，产生一个目标语言（英语）的句子B。

![img](https://pic1.zhimg.com/80/v2-7668b1c418d7ae79e043817a0eac3ac0_720w.webp)

图1. Transformer的功能示例，例子：输入是一句法语：Je（我）；suis（是）；etudiant（学生）。输出一个英文句子：I am a student。

打开这个黑盒子模型，我们可以看到它的内部结构，一个编码组件（encoding component），一个解码组件（decoding component）以及它们之间的连接。

![img](https://pic2.zhimg.com/80/v2-7acf38b210da6ecef71667c28dfc87ad_720w.webp)

图2. Transformer整体框架，包含一个编码组件，一个解码组件和他们之间的连接关系。

编码组件是由多个编码器（encoder）堆叠而来（原文中采用了6个编码器，6这个数字没有什么魔力，属于超参数，你在自己的实验中试验其他的数字如，2，4，8等），解码组件也是由相同个数的解码器堆叠而来。

![img](https://pic4.zhimg.com/80/v2-be39f5701a2787e93d5540c558feaf33_720w.webp)

图3. 编码组件和解码组件的展开示意图

所有的编码器在结构上是一样的，但是他们不共享权重。每一个编码器可以分为两个子层：

![img](https://pic1.zhimg.com/80/v2-ceed392dbcf0f7fe979da32a3ab757c8_720w.webp)

图4. 编码器展开示意图

编码器的输入首先流入一个self-attention层——该层可以帮助编码器在对特定单词进行编码时查看输入句子中的其他单词。稍后，我们将进一步了解self-attention。

self-attention层的输出被输入到前馈神经网络（feed-forward neural network，ffnn）。完全相同的前馈网络独立应用于每个位置（每一个单词向量都会通过这个ffnn层）。

解码器也具有这两层，但是在它们之间是一个attention层，可以帮助解码器关注到输入语句的相关部分上（类似于[seq2seq模型](https://link.zhihu.com/?target=https%3A//jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)中的attention）。

![img](https://pic3.zhimg.com/80/v2-1e8ec030be5be1c0ca7e90a8212e376a_720w.webp)

图5. 解码器结构图

## **把张量画出来**

我们已经了解了模型的主要组件，现在来关注一下模型中的各种向量/张量以及他们是如何在这些组件中流动，使得输入变成输出的。

一般地，在NLP任务中，我们首先需要通过[Embedding算法](https://link.zhihu.com/?target=https%3A//medium.com/deeper-learning/glossary-of-deep-learning-word-embedding-f90c3cec34ca)（嵌入算法）将输入中的每一个单词都转化为向量（word Embedding 单词嵌入）。

![img](https://pic3.zhimg.com/80/v2-226c51fe49f5d580c0554d4820df362e_720w.webp)

图6. 每个单词都被embed成为一个向量，长度为512，。我们通过这些简单的小方格来表示向量。

Embedding仅发生在最底部的编码器中（一般来说，Embedding是整个算法的第一个步骤）。所有编码器的共同点是，它们接收一个向量列表，每个向量的大小为512（在底部编码器（也就是第一个编码器）中是单词嵌入，但在其他编码器中，它们将接收下面的编码器的输出）。该列表的长度和列表中每一个向量的长度是我们可以设置的超参数，基本上，列表的长度就是训练数据集（或者是一次batch）中最长句子的长度。

将输入句子中的单词都进行单词嵌入以后，它们中的每一个都会流经编码器的两层。

![img](https://pic1.zhimg.com/80/v2-24097950f65ab3ee03c67b9bb77d4804_720w.webp)

图7. 单词嵌入输入到第一层（最底层）的编码器中

在这里，我们开始看到Transformer的一个关键属性，即每个位置的单词都流经编码器中自己的路径。 self-attention层中这些路径之间存在依赖关系。 但是，前馈层（Feed Forward）不具有这些依赖性，因此这些路径在流经前向网络时可以并行执行。

接下来，我们将示例切换到较短的句子，然后看一下编码器每个子层中发生的情况。

## **现在，我们开始编码！（Now We’re Encoding!）**

正如我们之前提到的，一个编码器接收一个向量列表作为输入。它通过将这些向量传递到“self-attention”层，然后传递到前馈神经网络，然后将其输出向上发送到下一个编码器。

![img](https://pic1.zhimg.com/80/v2-680f0c66b8e7c251d0950d39adfa871c_720w.webp)

图8. 每个位置的单词都会经过一个self-attention的过程。 然后，它们每个都通过前馈神经网络（图中的两个ffnn是完全相同的网络，每个向量分别流过它）。

## **在高层次中的Self-Attention（Self-Attention at a High Level）**

不要被“self-attention”这个词迷惑了，（同时自己又不怎么理解的时候）而觉得自己傻傻的，因为这个词不是每个人都应该熟悉的概念。 在阅读《Attention is All You Need》论文之前，我个人从未遇到过这个概念。 让我们提炼它是如何工作的。

假设我们要翻译以下句子：

”`The animal didn't cross the street because it was too tired`”

句中的“it”到底指代为何？是指街道还是动物？ 对于人类来说，这是一个简单的问题，但对于算法而言却并非如此简单。当模型在处理“it”的时候，self-attention可以将“it”和“animal”联系起来。

在模型处理每个单词（输入序列中的每个位置）时，self-attention使其能够查看输入序列中的其他位置以寻找线索，从而有助于更好地对该单词进行编码。

如果您熟悉RNN，请考虑一下如何通过保持隐藏状态来使RNN将其已处理的先前单词/向量的表示形式与当前正在处理的单词/向量进行合并。 **self-attention是Transformer用来将其他相关单词的“理解”融入我们当前正在处理的单词的方法**。

![img](https://pic3.zhimg.com/80/v2-d9b635c78e4d6728c0af655df5b1a182_720w.webp)

图9. 当我们在编码器5（堆栈中的顶部编码器）中对单词“ it”进行编码时，注意力机制的一部分集中在“ The Animal”上，并将其表示的一部分烘焙（融合）到“ it”的编码中。

在使用[Tensor2Tensor笔记本](https://link.zhihu.com/?target=https%3A//colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb)，您可以在其中加载Transformer模型，并使用此交互式可视化文件对其进行检查。

## **Self-Attention中的具体细节（Self-Attention in Detail）**

首先，让我们看一下self-attention中的向量计算过程，然后再着眼于如何在实际中使用矩阵来实现。

计算self-attention的第一步是从编码器的每个输入向量（在这种情况下，是每个单词的嵌入）创建三个向量。 因此，对于每个单词的嵌入向量，我们创建一个Query向量，一个Key向量和一个Value向量。 通过将嵌入向量乘以三个不同的矩阵来创建这些向量，这些矩阵会在训练过程中被逐渐优化。

请注意，这些新向量的维数小于嵌入向量的维数。 它们的维数为64（它们可大可小，论文中采用64是为了使多头注意力（大部分）计算保持恒定的体系结构），而嵌入和编码器输入/输出矢量的维数为512。

![img](https://pic3.zhimg.com/80/v2-2dd85f04f0a3a6c6577463b38528ed46_720w.webp)

图10. x1乘以WQ权重矩阵将得出q1，即与该单词关联的“Query”向量。 我们最终为输入句子中的每个单词创建一个“Query”，一个“Key”和一个“Value”向量。

“Query”，“Key”，“Value”到底是什么呢？

它们是抽象，对于计算和思考注意力非常有用。 继续阅读下面的注意力计算方式后，您将了解所有这些向量所起的作用。

计算self-attention的第二步是计算分数。 假设我们正在执行示例中第一个单词“Thinking”的自注意力计算路劲。 我们需要根据“Thinking”对输入句子的每个单词进行评分。 分数决定了当我们更新单词“Thinking”的编码时，将注意力集中在输入句子的其他部分上的程度（其他单词对“Thinking”编码的贡献程度）。

在编码阶段针对单词A进行编码时，需要计算输入句子中每一个单词（包括A）与单词A的注意力分数。当我们在计算单词B和单词A之间的注意力分数时，通过将单词A的query向量与单词B的key向量进行点积即可得出的。 因此，如果我们正在处理位置＃1上的单词的自注意，则第一个分数将是q1和k1的点积。 第二个分数是q1和k2的点积，以此类推。

![img](https://pic4.zhimg.com/80/v2-3387b365747bae029c9bbd34ec323beb_720w.webp)

图11. 注意力分数的计算

第三和第四步是将分数除以8（本文使用的key向量的维数64的平方根。这将导致具有更稳定的渐变。此处可能存在其他可能的值，但这是默认值），然后通过softmax函数来对分数进行归一化，使它们均为正，且和为1。

![img](https://pic4.zhimg.com/80/v2-6ade838b6b62f38cbfac50bdbd08482b_720w.webp)

图12. 将注意力分数除以8，并出入到Softmax函数中进行处理

经过softmax之后的分数确定每个单词在此位置（在图12中的例子上为第一个位置“Thinking”）的编码上将被表达多少。 显然，此位置的单词本身的softmax得分最高，但是有时会对寻找与当前单词相关的另一个单词很有用。

第五步是将每个value向量乘以对应的softmax分数（准备对其求和）。 直觉是保持我们要关注的单词的值完整，并淹没无关的单词（例如，将它们乘以0.001之类的小数字）也就是保留关注词的value值，削弱非相关词的value值。。

第六步是对value向量进行求和加权。 这将在此位置（对于第一个单词）产生self-attention层的输出。

![img](https://pic3.zhimg.com/80/v2-84a6da831e4ab647caa7bd4029b41daa_720w.webp)

图13. self-attention的输出：所有Value向量的加权求和

这样就完成了self-attention的计算。 生成的向量是我们可以发送到前馈神经网络的向量。 但是，在实际实现中，此计算以矩阵的形式进行，以加快处理速度。 现在，让我们来看一下是如何执行矩阵计算的。

## Self-Attention的矩阵计算（**Matrix Calculation of Self-Attention**）

第一步是计算Query，Key和Value矩阵。 为此，我们将嵌入向量打包为矩阵X，然后将其乘以我们的可训练权重矩阵（ �� ， �� ， �� ）：

![img](https://pic1.zhimg.com/80/v2-dec2a5aeac531a1d0e56548924999ed0_720w.webp)

图14. 矩阵上的Query，Key和Value计算

由于我们要处理矩阵，因此我们可以将步骤2到6压缩成一个公式，以计算Self-Attention层的输出。

![img](https://pic2.zhimg.com/80/v2-b4e66144f21b39b9706c465926c05ded_720w.webp)

图15. 矩阵运算上的Softmax操作

## 最好采用多头注意力（**The Beast With Many Heads**）

该论文通过添加一种称为“多头”注意力的机制，进一步完善了self-attention层。 该操作在一下两个方面提高attention层的性能：

1. 它扩展了模型专注于不同位置的能力。在上面的示例中，z1主要由自己本身决定，只包含所有其他编码的一点点信息，有时候胡出现信息不足，模型效果不好的情况，比如我们要翻译这样的句子，“The animal didn’t cross the street because it was too tired”，那么我们会想知道“it”指的是什么，那么多头注意力就有用了，它可以从不同的方面出发，关注不同的词汇，比较完整的解出“it”所涉及的单词。
2. 它为关注层提供了多个“表示子空间”。 正如我们接下来将要看到的，在多头关注下，我们不仅拥有一个而是多组Query/Key/Value权重矩阵（Transformer使用八个关注头，因此每个编码器/解码器最终得到八组输出） 。 这些权重矩阵中的每一个都是随机初始化的。 然后，在训练之后，将每个权重矩阵用于将输入的嵌入（或来自较低编码器/解码器的输出矩阵）投影到不同的表示子空间中。

![img](https://pic4.zhimg.com/80/v2-86240fbafc75246e685487f41f9a8903_720w.webp)

图16. 在多头关注下，我们为每个头维护单独的Q / K / V权重矩阵，从而导致不同的Q / K / V矩阵。 如前所述，我们将X乘以WQ / WK / WV矩阵以生成Q / K / V矩阵。

如果我们执行上面概述的多头自注意力计算，经过八组不同的权重矩阵，最终将得到八个不同的Z矩阵。

![img](https://pic1.zhimg.com/80/v2-edc0eb907916fe40ced24e706e10b7f8_720w.webp)

图17. 多头注意力机制下产生多个输出Z

这给我们带来了一些挑战。 前馈层不希望有8个矩阵，而是一个矩阵（每个单词一个向量）。 因此，我们需要一种将这八个矩阵压缩为单个矩阵的方法。

我们该怎么做？ 我们合并（拼接）矩阵，然后将它们乘以权重矩阵 �� 。

![img](https://pic4.zhimg.com/80/v2-505447e0e434ab493a21611625f21b07_720w.webp)

图18. 将多个输出Z压缩为一个书输出Z

这就是多头注意力的全部。 我知道，矩阵很多。 让我尝试将它们全部放在一个图片中，以便我们可以在一处查看它们。

![img](https://pic4.zhimg.com/80/v2-289ebc95a9eeaf7ff2b13850afa5a717_720w.webp)

图19. Self-Attention完整矩阵计算流流程图

既然我们已经涉及到attention head，那么让我们重新回顾一下示例，看看在示例句中对“ it”一词进行编码时，不同的attention head所关注的位置：

![img](https://pic4.zhimg.com/80/v2-ea5e10de7609e2e20208a0e8ffcdb28f_720w.webp)

图20. 当我们对“ it”一词进行编码时，一个注意力集中在“the animal”上，而另一个则集中在“tired”上-从某种意义上说，模型对单词“ it”的表示一方面体现在“the animal”上，另一方面体现在“tired”上。

但是，如果将所有attention head的结果添加到图片中，则可能很难解释：

![img](https://pic3.zhimg.com/80/v2-98ca506acbc9ddb8b08104b7e6a0c906_720w.webp)

图21. 所有attention head的可视化

## 使用位置编码表示序列的顺序（**Representing The Order of The Sequence Using Positional Encoding**）

到目前为止，我们描述的模型中缺少的一件事是一种解决输入序列中单词顺序的方法。（CNN和RNN都是顺序敏感的模型，他们的输出与单词的顺序有着紧密的关系，而Transformer对位置不敏感。）

为了解决这个问题，Transformer将位置向量添加到每个单词嵌入中。 这些向量遵循模型学习的特定模式，这有助于确定每个单词的位置或序列中不同单词之间的距离。 直觉是，将这些值添加到嵌入向量中后，一旦将它们投影到Q / K / V向量中，那么在点积乘法和注意分数计算期间，就可以在特征向量之间提供有意义的距离。

![img](https://pic3.zhimg.com/80/v2-c08457499090a096e71bbc027d37e42a_720w.webp)

图22. 为了使模型可以感知单词的顺序，我们添加了位置编码向量（位置编码向量遵循特定的模式）。

假设嵌入的维数为4，则实际的位置编码应如下所示：

![img](https://pic2.zhimg.com/80/v2-df5f90e0b643d3c5569cdc73fdd3d8d1_720w.webp)

图23. 嵌入大小为4的位置编码的真实示例

这种模式是什么样的？

在下图中，每行对应一个向量的位置编码。 因此，第一行将是我们要添加到输入序列中第一个单词的嵌入的向量。 每行包含512个值（每个值都在1到-1之间）。 我们对它们进行了颜色编码，以便可视化。

![img](https://pic2.zhimg.com/80/v2-a8b7e034b43892d1fbdfd1b767c8d9c1_720w.webp)

图24. 嵌入大小为512（列）的20个单词（行）的位置编码的真实示例。 您会看到它看起来像是在中心处分成两半。 这是因为左半部分的值是由一个函数（使用正弦）生成的，而右半部分的值是由另一个函数（使用余弦）生成的。 然后将它们串联起来以形成每个位置编码向量。

在论文的3.5节中描述了位置编码的公式。 您可以在[get_timing_signal_1d（）](https://link.zhihu.com/?target=https%3A//github.com/tensorflow/tensor2tensor/blob/23bd23b9830059fbc349381b70d9429b5c40a139/tensor2tensor/layers/common_attention.py)中看到用于生成位置编码的代码。 这不是位置编码的唯一可能方法。但是，无论采用何种编码方法，该算法必须能够处理未知长度的序列。（例如，如果我们训练好的模型要求翻译的句子比训练集中的任何句子更长）。

上面显示的位置编码来自Transformer的Tranformer2Transformer实现。 论文中提供的方法略有不同，因为它不直接串联，而是交织两个信号。 下图显示了它是什么样子的。 [这是生成它的代码](https://link.zhihu.com/?target=https%3A//github.com/jalammar/jalammar.github.io/blob/master/notebookes/transformer/transformer_positional_encoding_graph.ipynb)：

![img](https://pic4.zhimg.com/80/v2-cfe0607308065f530008ce9be6308367_720w.webp)

图25. 论文《attention is all you need》中位置编码的可视化示意图

## **残差结构（The Residuals）**

在继续进行之前，我们需要提到的编码器体系结构中的一个细节是，每个编码器中的每个子层（Self-Attention，ffnn）在其周围都有残差连接与层归一化（[layer normalization](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1607.06450)）操作。

![img](https://pic4.zhimg.com/80/v2-c7160ee471946af4b0f59bb376e88e13_720w.webp)

图26. 完整的编码器结构，包含self-attention，Feed forward，残差连接，layer normalization

如果我们要可视化向量与self-attention层相关的layer normalization操作，则看起来应该像这样：

![img](https://pic3.zhimg.com/80/v2-8c1cf28ef4e3243eaad49df334bba51e_720w.webp)

图27. 结合向量可视化的编码器流程图

这也适用于解码器的子层。 假设Transformer由2个堆叠式编码器和解码器组成的，它看起来像这样：

![img](https://pic3.zhimg.com/80/v2-5841f646b25f60335321c34960c57bda_720w.webp)

图28. 两层编码器与两层解码器的Transformer结构示意图

## **解码器结构（The Decoder Side）**

现在，我们已经介绍了编码器方面的大多数概念，我们也基本了解码器的组件的工作流程。 但是，让我们看看编码器和解码器是如何协同工作的。

编码器首先处理输入序列。 然后，顶部编码器的输出转换为注意向量K和V的集合。每个解码器将在其“编码器-解码器注意”层中使用它们，这有助于解码器中在输入序列中关注适当位置：

![动图封面](https://pic2.zhimg.com/v2-f457bd6dec5913355eb082912f657885_b.jpg)



图29. 编码器解码器协同工作流程示意图，完成编码阶段后，我们开始解码阶段。 解码阶段的每个步骤都会从输出序列中输出一个元素（图示中为法语翻译为英语）。

以下步骤重复该过程，直到输出一个特殊符号，该符号指示transformer解码器已完成其输出。 每个步骤的输出将在下一个时间步骤中馈送到底部解码器，同时我们在这些解码器输入中嵌入并添加位置编码以指示每个单词的位置。

![动图封面](https://pic2.zhimg.com/v2-f234ea0b61cc6c155581533add4396c9_b.jpg)



图30. 解码器运作流程图

解码器中的self-attention层与编码器中的self-attention层略有不同：

在解码器中，仅允许self-attention层参与输出序列中的较早位置。 这是通过在self-attention计算中的softmax步骤之前屏蔽将来的位置（将它们设置为-inf）来完成的。

“ Encoder-Decoder Attention”层的工作方式与多头自注意力类似，不同之处在于它从其下一层创建其Queries矩阵，并从编码器堆栈的输出中获取Keys和Values矩阵。

## 最后的线性层与Softmax层（**The Final Linear and Softmax Layer**）

解码器堆栈输出浮点向量。 我们如何把它变成一个词？ 这就是The Final Linear后接Sotomax的工作了

线性层是一个简单的完全连接的神经网络，它将解码器堆栈产生的向量投影到一个更大的向量中，称为logits向量。

假设我们的目标语言中共有10,000个单词。那么线性层的输出维度（logits向量的宽度） 为10,000个单元（每个单元对应一个唯一单词的分数）。 这就是我们解释模型的输出以及线性层的方式。

然后，softmax层将这些分数转换为概率（全部为正，全部相加为1.0）。 选择具有最高概率的单元，然后该单元对应的单词将作为该时间步的输出。

![img](https://pic1.zhimg.com/80/v2-aad5957b94dee20c4108c3982b6e9838_720w.webp)

图31. 该图从底部开始，将产生的向量作为解码器堆栈的输出。 然后将其转换为对应的word。

## 训练回顾（**Recap Of Training**）

现在，我们已经了解了一个训练完毕的Transformer的前向过程的具体细节，顺便看下与训练相关的内容也是非常有用的。

在训练过程中，未经训练的模型将经历上述的前向过程。 当我们在带标签的训练集上训练时，可以对比预测输出与实际输出。

为了直观地说明这一点，我们假设输出词汇表仅包含六个单词（“ a”，“ am”，“ i”，“ thanks”，“ student”和“ <eos>”（“end of sentence”的缩写）） 。

![img](https://pic2.zhimg.com/80/v2-354a133a62d73c4eed7f546f1c42ebdd_720w.webp)

图32. 我们的模型的输出词汇表是在预处理阶段创建的，甚至在没有开始训练之前就定义好了。

一旦定义了输出词汇表，我们就可以使用相同宽度的向量来指示词汇表中的每个单词。 这也称为独热编码。例如，我们可以使用以下向量表示单词“ am”：

![img](https://pic4.zhimg.com/80/v2-a6fad654b10d3453cb3a5b1a000c7443_720w.webp)

图33. 一个输出词汇的独热编码的例子

接下来让我们讨论模型的损失函数，我们在训练阶段优化该函数，可以得出经过训练的，非常惊人的准确模型。

## 损失函数（**The Loss Function**）

假设我们正在训练我们的模型，这是我们训练阶段的第一步，我们将以一个简单的示例来展示训练过程，将“ merci”转换为“thanks”。

这意味着我们希望输出是一个表示单词“thanks”的概率分布。 但是，由于尚未对该模型进行训练，因此目前不太可能发生。

![img](https://pic1.zhimg.com/80/v2-234cb762c294266a891db298ddd1c974_720w.webp)

图34. 如果模型的参数（权重）都是随机初始化的，那么（未经训练的）模型会为每个像元/单词生成具有任意值的概率分布。 我们可以将其与实际输出进行比较，然后使用反向传播调整所有模型的权重，以使输出更接近所需的输出。

我们如何比较两个概率分布？ 我们简单地从另一个中减去一个。 更多相关详细信息，请查看[交叉熵](https://link.zhihu.com/?target=https%3A//colah.github.io/posts/2015-09-Visual-Information/)和[Kullback-Leibler散度](https://link.zhihu.com/?target=https%3A//www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained)。

但是请注意，这是一个过于简化的示例。 实际上，我们将使用一个不止一个单词的句子。 例如，输入：“ je suis étudiant”，预期输出：“i am a student”。 这实际上意味着我们期望模型输出连续的概率分布满足如下条件：

- 每个概率分布都由一个宽度为vocab_size的向量表示（在我们的示例中为vocab_size=6，但是更实际地为30,000或50,000）
- 第一个概率分布在与单词“ i”相关联的单元中具有最高概率
- 第二个概率分布在与单词“ am”相关联的单元格处具有最高概率
- 依此类推，直到第五个输出分布指示“ <end of sentence>”符号为止，在我们定义的词汇表中，该符号也具有相关的单元格。

![img](https://pic4.zhimg.com/80/v2-4500fa28495ea7126393dacae67ec1bf_720w.webp)

表35. 依据例子训练模型得到的目标概率分布

在足够大的数据集上训练模型足够的时间后，我们希望产生的概率分布如下所示：

![img](https://pic2.zhimg.com/80/v2-75eab5bc3bed7b33bc31159bc240ed49_720w.webp)

图36. 我们期望训练过后，模型会输出正确的翻译。如果这段话完全来自训练集，它并不是一个很好的评估指标（参考交叉验证，https://www.youtube.com/watch?v=TIgfjmp-4BA）。注意到每个位置都得到了一点概率，即使它不太可能成为那个时间步的输出——这是softmax的一个很有用的性质，它可以帮助训练模型。

现在，由于该模型一次生成一个输出，我们可以假定该模型从该概率分布中选择具有最高概率的单词，然后丢弃其余单词。 这是其中一种方法（称为贪婪解码）。 另一种方法是坚持使用前两个单词（例如，“ I”和“ a”），然后在下一步中运行模型两次：在第一次中假设第一个输出位置为 单词“ I”，另一次中假设第一个输出位置是单词“ a”，并且无论哪个版本产生更少的误差，都保留概率最高的两个翻译结果。 我们在位置2和位置3等重复此操作。 此方法称为集束搜索（beam search），在我们的示例中，beam_size为两个（意味着始终在内存中保留两个部分假设（未完成的翻译）），并且top_beams也为两个（意味着我们将返回两个翻译 ）。 这些都是您可以尝试的超参数。

## **更进一步然后改进（Go Forth And Transform）**

希望您已经找到了一个有用的地方来开始用Transformer的主要概念来打破僵局。 如果您想更进一步，建议您执行以下步骤：

- 阅读论文 [Attention Is All You Need](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1706.03762) , 博客 ([Transformer: A Novel Neural Network Architecture for Language Understanding](https://link.zhihu.com/?target=https%3A//ai.googleblog.com/2017/08/transformer-novel-neural-network.html)), 和 [Tensor2Tensor announcement](https://link.zhihu.com/?target=https%3A//ai.googleblog.com/2017/06/accelerating-deep-learning-research.html).
- 观看 [Łukasz Kaiser’s talk](https://link.zhihu.com/?target=https%3A//www.youtube.com/watch%3Fv%3DrBCqOTEfxvg) 来了解模型及其细节
- 操作 [Jupyter Notebook provided as part of the Tensor2Tensor repo](https://link.zhihu.com/?target=https%3A//colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb)
- 尝试一下项目： [Tensor2Tensor repo](https://link.zhihu.com/?target=https%3A//github.com/tensorflow/tensor2tensor).

后续工作：

- [Depthwise Separable Convolutions for Neural Machine Translation](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1706.03059)
- [One Model To Learn Them All](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1706.05137)
- [Discrete Autoencoders for Sequence Models](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1801.09797)
- [Generating Wikipedia by Summarizing Long Sequences](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1801.10198)
- [Image Transformer](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1802.05751)
- [Training Tips for the Transformer Model](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1804.00247)
- [Self-Attention with Relative Position Representations](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1803.02155)
- [Fast Decoding in Sequence Models using Discrete Latent Variables](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1803.03382)
- [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1804.04235)