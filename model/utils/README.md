# utils 文件夹的文件使用方法
这篇文档位于utils包下，是整个我写的models里面最关键的部分，其他的模型只不过是在这些基础上对forward层进行了更改，
这里封装了很多函数。

## 1. BaseModel

该模块中编译了所有推荐模型的父模型。它通过调用DataRecorder对象来获取模型输入数据的全部信息。该模块包含模型的：
1. 初始化部分：目前仅有He氏初始化。暂未加入Xavier初始化方法。
2. 优化器：默认使用AdamW优化器，还可选择Adam和SGD
3. 损失函数使用BCE函数
4. 其他参数默认值：学习率0.01，批次大小256，embedding维度12，早停容忍度5

在模型训练的时候，我们为了加快收敛速度，对DataLoader中的参数也引入了并行化参数`num_workers=8`, `pin_memory=True`,
这样可以让8个线程同时加载数据，并加载到GPU中实现。

## 2. Layers

这里封装了模型常用的一些层的类。

### 1. MLP多层感知机
包括普通的MLP和带残差链接的MLP，这两个比较简单，本文对于MLP的调参也进行了很多，加入了dropout层控制参数，以及层归一化或列归一化的控制选项。这里比较简单，就不再赘述。


### 2. LinearRegression Layer
对于LinearRegression层的作用，为什么不用`nn.Linear`，反而自己定义一个类？这是作者从FM类模型的`实战经验中得出的`，例如Wide&Deep模型，Deep模型的输出结果已经被充分的缩放和归一化，其尺度与未经过embedding，直接接入线性层的数据`量纲相差极大`，因此，需要对Wide层，也就是线性层的输出进行归一化处理。这就是我们自定义的线性层的作用，此外，这里的实现方法并非先输出结果在归一化，而是以一种巧妙的方法：我们在将`DataRecorder`类，建立`Embedding Schema`的时候额外储存一个映射字典，保存将各个输入特征映射到一维的字典`nn.Embedding(input_dim, 1)`，我们再将输出结果加和得到输出。这种实际上就是对输入特征做了一步 $\Sigma w_i x_i$ 也就是线性回归，这种方法不仅处理了量纲问题，且收敛更快。

### 3. Embedding Layer
对于Embedding层，普通的Embedding层就是按图索骥，按照记录的Embedding方案去做就好了。

此外，本模型根据华为的 [AutoDis](../../Document/特征离散化/AutoDis.pdf) 模型，复现了他对于连续变量的自动处理方法。

## 3. [LargeScalePreprocess.py](LargeScalePreprocess.py)

DataFrame会将所有输出存到内存，且HDFS数据集下载下来的通常存为Avro格式，因此我们需要开发一套数据处理的API来处理大规模数据，使其在GPU上能高效运行。
本仓库选取 `Parquet` 作为数据结构，它保留了表格的操作，同时可以高效地操作数据。

### `Parquet` 类基本用法
本仓库的数据处理阶段应用的都是Parquet数据结构，他处理几十GB的数据时效果显著。

1. `parquet_table = pq.ParquetFile(parquet_file)`： 它是 PyArrow 库中用于读取和写入 Parquet 文件的一个类。当你想要读取一个 Parquet 文件时，可以使用 ParquetFile 类来打开文件，然后使用其提供的方法来读取数据。
2. `parquet_table.metadata` 是表的元信息，里面包含了：
   - `.num_columns` 列的个数
   - `.num_rows` 行的个数
   - `.num_row_groups` 分区的个数，这个取决于你读取parquet文件的时候是怎么操作的
3. `parquet_table.schema` 是表头信息，他是一个对象，只有一个功能就是再用`parquet_table.schema.names`访问列的名字。 返回值是一个 `list` 对象。

4. `parquet_table.read_row_group` 用于分group读取每个group的信息。 返回的结果是一个 `PyArrow Table` 格式的数据。 然后再使用 `.column(0)` 返回第一列，然后再 `.to_pylist() `转成`list`格式

本仓库主要使用第四条，由于数据集庞大，无法一次性加载到GPU，且这样效率不高，因此分片读入是很好的选择。

此外，本仓库为了加速训练做了很多调试，在对特征进行encode的时候，本仓库采用了多线程的操作，测试所用的处理器为`8核16线程`，因此，本仓库中将其设为8个worker共同运作，在每个子线
程中计算编码。具体实现方法是将子线程写作一个`静态方法`传入主线程。
