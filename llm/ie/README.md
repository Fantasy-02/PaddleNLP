# 通用信息抽取 UIE(Universal Information Extraction)

 **目录**

- [1. 模型简介](#模型简介)
- [2. 应用示例](#应用示例)
- [3. 开箱即用](#开箱即用)
  - [3.1 实体抽取](#实体抽取)
  - [3.2 关系抽取](#关系抽取)
  - [3.3 事件抽取](#事件抽取)
  - [3.4 模型选择](#模型选择)
  - [3.5 更多配置](#更多配置)
- [4. 训练定制](#训练定制)
  - [4.1 代码结构](#代码结构)
  - [4.2 数据标注](#数据标注)
  - [4.3 模型微调](#模型微调)
  - [4.4 模型推理](#模型推理)
  - [4.5 定制模型一键预测](#定制模型一键预测)
  - [4.6 模型快速服务化部署](#模型快速服务化部署)
  - [4.7 实验指标](#实验指标)
  - [4.8 模型部署](#模型部署)

<a name="模型简介"></a>

## 1. 模型简介

Yaojie Lu 等人在 ACL-2022中提出了通用信息抽取统一框架 UIE。该框架实现了实体抽取、关系抽取、事件抽取、情感分析等任务的统一建模，并使得不同任务间具备良好的迁移和泛化能力。然而，该模型在零样本场景下的表现仍存在不足。为此，PaddleNLP 借鉴 UIE 的方法，基于 Qwen2.5-0.5B 预训练模型，训练并开源了一款面向中文通用信息抽取的大模型。

<!-- <div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/167236006-66ed845d-21b8-4647-908b-e1c6e7613eb1.png height=400 hspace='10'/>
</div> -->



<!-- #### UIE 的优势

- **使用简单**：用户可以使用自然语言自定义抽取目标，无需训练即可统一抽取输入文本中的对应信息。**实现开箱即用，并满足各类信息抽取需求**。

- **降本增效**：以往的信息抽取技术需要大量标注数据才能保证信息抽取的效果，为了提高开发过程中的开发效率，减少不必要的重复工作时间，开放域信息抽取可以实现零样本（zero-shot）或者少样本（few-shot）抽取，**大幅度降低标注数据依赖，在降低成本的同时，还提升了效果**。

- **效果领先**：开放域信息抽取在多种场景，多种任务上，均有不俗的表现。

<a name="应用示例"></a> -->

## 2. 应用示例



<a name="开箱即用"></a>

## 3. 开箱即用

```paddlenlp.Taskflow```提供通用信息抽取、评价观点抽取等能力，可抽取多种类型的信息，包括但不限于命名实体识别（如人名、地名、机构名等）、关系（如电影的导演、歌曲的发行时间等）、事件（如某路口发生车祸、某地发生地震等）、以及评价维度、观点词、情感倾向等信息。用户可以使用自然语言自定义抽取目标，无需训练即可统一抽取输入文本中的对应信息。**实现开箱即用，并满足各类信息抽取需求**

<a name="实体抽取"></a>

#### 3.1 实体抽取

  命名实体识别（Named Entity Recognition，简称 NER），是指识别文本中具有特定意义的实体。在开放域信息抽取中，抽取的类别没有限制，用户可以自己定义。

  - 例如抽取的目标实体类型是"时间"、"选手"和"赛事名称", schema 构造如下：

    ```text
    ['时间', '选手', '赛事名称']
    ```

    调用示例：

    ```python
    from pprint import pprint
    from paddlenlp import Taskflow

    schema = ['时间', '选手', '赛事名称'] # Define the schema for entity extraction
    ie = Taskflow('information_extraction',
                  schema= ['时间', '选手', '赛事名称'],
                  schema_lang="zh",
                  batch_size=1,
                  model='qwen-0.5b'）
    pprint(ie("2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！")) # Better print results using pprint
    # 输出
    [{'时间': [{'text': '2月8日上午'}],
      '赛事名称': [{'text': '北京冬奥会自由式滑雪女子大跳台决赛'}],
      '选手': [{'text': '谷爱凌'}]}]
    ```


<a name="关系抽取"></a>

#### 3.2 关系抽取

  关系抽取（Relation Extraction，简称 RE），是指从文本中识别实体并抽取实体之间的语义关系，进而获取三元组信息，即<主体，谓语，客体>。
  
  - 例如以"竞赛名称"作为抽取主体，抽取关系类型为"主办方"、"承办方"和"已举办次数", schema 构造如下：

    ```text
    {
      '竞赛名称': [
        '主办方',
        '承办方',
        '已举办次数'
      ]
    }
    ```

    调用示例：

    ```python
    schema = {'竞赛名称': ['主办方', '承办方', '已举办次数']} # Define the schema for relation extraction
    ie.set_schema(schema) # Reset schema
    pprint(ie('2022语言与智能技术竞赛由中国中文信息学会和中国计算机学会联合主办，百度公司、中国中文信息学会评测工作委员会和中国计算机学会自然语言处理专委会承办，已连续举办4届，成为全球最热门的中文NLP赛事之一。'))
    # 输出
    [{'竞赛名称': [{'relations': {'主办方': [{'text': '中国中文信息学会,中国计算机学会'}],
                          '已举办次数': [{'text': '4'}],
                          '承办方': [{'text': '百度公司,中国中文信息学会评测工作委员会,中国计算机学会自然语言处理专委会'}]},
            'text': '2022语言与智能技术竞赛'}]}]
    ```

<a name="事件抽取"></a>

#### 3.3 事件抽取

  事件抽取 (Event Extraction, 简称 EE)，是指从自然语言文本中抽取预定义的事件触发词(Trigger)和事件论元(Argument)，组合为相应的事件结构化信息。

  - 例如抽取的目标是"地震"事件的"地震强度"、"时间"、"震中位置"和"震源深度"这些信息，schema 构造如下：

    ```text
    {
      '地震触发词': [
        '地震强度',
        '时间',
        '震中位置',
        '震源深度'
      ]
    }
    ```

    触发词的格式统一为`触发词`或``XX 触发词`，`XX`表示具体事件类型，上例中的事件类型是`地震`，则对应触发词为`地震触发词`。

    调用示例：

    ```python
    >>> schema = {'地震触发词': ['地震强度', '时间', '震中位置', '震源深度']} # Define the schema for event extraction
    >>> ie.set_schema(schema) # Reset schema
    >>> ie('中国地震台网正式测定：5月16日06时08分在云南临沧市凤庆县(北纬24.34度，东经99.98度)发生3.5级地震，震源深度10千米。')
    [{'地震触发词': [{'text': '地震', 'relations': {'地震强度': [{'text': '3.5级'}], '时间': [{'text': '5月16日06时08分'}], '震中位置': [{'text': '云南临沧市凤庆县(北纬24.34度，东经99.98度)'}], '震源深度': [{'text': '10千米'}]}}]}]
    ```



#### 3.4 模型选择

- 多模型选择，满足精度、速度要求
<!-- 
  | 模型 |  结构  | 语言 |
  | :---: | :--------: | :--------: |
  | `uie-base` (默认)| 12-layers, 768-hidden, 12-heads | 中文 |
  | `uie-base-en` | 12-layers, 768-hidden, 12-heads | 英文 |
  | `uie-medical-base` | 12-layers, 768-hidden, 12-heads | 中文 |
  | `uie-medium`| 6-layers, 768-hidden, 12-heads | 中文 |
  | `uie-mini`| 6-layers, 384-hidden, 12-heads | 中文 |
  | `uie-micro`| 4-layers, 384-hidden, 12-heads | 中文 |
  | `uie-nano`| 4-layers, 312-hidden, 12-heads | 中文 |
  | `uie-m-large`| 24-layers, 1024-hidden, 16-heads | 中、英文 |
  | `uie-m-base`| 12-layers, 768-hidden, 12-heads | 中、英文 | -->




#### 3.5 更多配置

```python
>>> from paddlenlp import Taskflow

>>> ie = Taskflow('information_extraction',
                  schema="",
                  schema_lang="zh",
                  batch_size=1,
                  model='qwen-0.5b',
                  position_prob=0.5,
                  precision='fp16',
                  use_fast=False)
```

* `schema`：定义任务抽取目标，可参考开箱即用中不同任务的调用示例进行配置。
* `schema_lang`：设置 schema 的语言，默认为`zh`, 可选有`zh`和`en`。因为中英 schema 的构造有所不同，因此需要指定 schema 的语言。该参数只对`uie-m-base`和`uie-m-large`模型有效。
* `batch_size`：批处理大小，请结合机器情况进行调整，默认为1。
* `model`：选择任务使用的模型，默认为`qwen-0.5b`，可选有`qwen-0.5b`, `qwen-1.5b`。
* `precision`：选择模型精度，默认为`fp32`，可选有`fp16`和`fp32`。`fp16`推理速度更快，支持 GPU 和 NPU 硬件环境。如果选择`fp16`，在 GPU 硬件环境下，请先确保机器正确安装 NVIDIA 相关驱动和基础软件，**确保 CUDA>=11.2，cuDNN>=8.1.1**，初次使用需按照提示安装相关依赖。其次，需要确保 GPU 设备的 CUDA 计算能力（CUDA Compute Capability）大于7.0，典型的设备包括 V100、T4、A10、A100、GTX 20系列和30系列显卡等。更多关于 CUDA Compute Capability 和精度支持情况请参考 NVIDIA 文档：[GPU 硬件与支持精度对照表](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-840-ea/support-matrix/index.html#hardware-precision-matrix)。
<a name="训练定制"></a>

## 4. 训练定制

对于简单的抽取目标可以直接使用 ```paddlenlp.Taskflow```实现零样本（zero-shot）抽取，对于细分场景我们推荐使用轻定制功能（标注少量数据进行模型微调）以进一步提升效果。下面通过`报销工单信息抽取`的例子展示如何通过5条训练数据进行 UIE 模型微调。

<a name="代码结构"></a>

#### 4.1 代码结构

```shell
.
├── utils.py          # 数据处理工具
├── doccano.py        # 数据标注脚本
├── doccano.md        # 数据标注文档
└── README.md
```

<a name="数据标注"></a>

#### 4.2 数据标注

我们推荐使用数据标注平台[doccano](https://github.com/doccano/doccano) 进行数据标注，本示例也打通了从标注到训练的通道，即 doccano 导出数据后可通过[doccano.py](./doccano.py)脚本轻松将数据转换为输入模型时需要的形式，实现无缝衔接。标注方法的详细介绍请参考[doccano 数据标注指南](doccano.md)。

原始数据示例：

```text
深大到双龙28块钱4月24号交通费
```

抽取的目标(schema)为：

```python
schema = ['出发地', '目的地', '费用', '时间']
```

标注步骤如下：

- 在 doccano 平台上，创建一个类型为``序列标注``的标注项目。
- 定义实体标签类别，上例中需要定义的实体标签有``出发地``、``目的地``、``费用``和``时间``。
- 使用以上定义的标签开始标注数据，下面展示了一个 doccano 标注示例：

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/167336891-afef1ad5-8777-456d-805b-9c65d9014b80.png height=100 hspace='10'/>
</div>

- 标注完成后，在 doccano 平台上导出文件，并将其重命名为``doccano_ext.json``后，放入``./data``目录下。

- 这里我们提供预先标注好的文件[doccano_ext.json](https://bj.bcebos.com/paddlenlp/datasets/uie/doccano_ext.json)，可直接下载并放入`./data`目录。执行以下脚本进行数据转换，执行后会在`./data`目录下生成训练/验证/测试集文件。

```shell
python doccano.py \
    --doccano_file ./data/doccano_ext.json \
    --save_dir ./data \
    --splits 0.8 0.2 0 \
    --schema_lang ch
```


可配置参数说明：

- ``doccano_file``: 从 doccano 导出的数据标注文件。
- ``save_dir``: 训练数据的保存目录，默认存储在``data``目录下。
- ``negative_ratio``: 最大负例比例，该参数只对抽取类型任务有效，适当构造负例可提升模型效果。负例数量和实际的标签数量有关，最大负例数量 = negative_ratio * 正例数量。该参数只对训练集有效，默认为5。为了保证评估指标的准确性，验证集和测试集默认构造全正例。
- ``splits``: 划分数据集时训练集、验证集所占的比例。默认为[0.8, 0.1, 0.1]表示按照``8:1:1``的比例将数据划分为训练集、验证集和测试集。
- ``task_type``: 选择任务类型，目前只有信息抽取这一种任务。
- ``is_shuffle``: 是否对数据集进行随机打散，默认为 True。
- ``seed``: 随机种子，默认为1000.
- ``schema_lang``: 选择 schema 的语言，可选有`ch`和`en`。默认为`ch`，英文数据集请选择`en`。

备注：
- 默认情况下 [doccano.py](./doccano.py) 脚本会按照比例将数据划分为 train/dev/test 数据集
- 每次执行 [doccano.py](./doccano.py) 脚本，将会覆盖已有的同名数据文件
- 在模型训练阶段我们推荐构造一些负例以提升模型效果，在数据转换阶段我们内置了这一功能。可通过`negative_ratio`控制自动构造的负样本比例；负样本数量 = negative_ratio * 正样本数量。
- 对于从 doccano 导出的文件，默认文件中的每条数据都是经过人工正确标注的。


<a name="模型微调"></a>

#### 4.3 模型微调

推荐使用 [大模型精调](../docs/finetune.md) 对模型进行微调。只需输入模型、数据集等就可以高效快速地进行微调和模型压缩等任务，可以一键启动多卡训练、混合精度训练、梯度累积、断点重启、日志显示等功能，并且针对训练过程的通用训练配置做了封装，比如：优化器、学习率调度等。

使用下面的命令，使用 `qwen-0.5B` 作为预训练模型进行模型微调，将微调后的模型保存至`$finetuned_model`：

如果在 GPU 环境中使用，可以指定 gpus 参数进行多卡训练：

```shell
# 返回到llm目录下
cd ..
python -u  -m paddle.distributed.launch --gpus "0,1" run_finetune.py ./config/qwen/sft_argument.json
```

sft_argument.json的参考配置如下：
```shell
{
    "model_name_or_path": "Qwen/Qwen2.5-0.5B",
    "dataset_name_or_path": "./ie/data",
    "output_dir": "./checkpoints/sft_ckpts",
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "per_device_eval_batch_size": 1,
    "eval_accumulation_steps":8,
    "num_train_epochs": 3,
    "learning_rate": 3e-05,
    "warmup_steps": 30,
    "logging_steps": 1,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "src_length": 1024,
    "max_length": 2048,
    "bf16": false,
    "fp16_opt_level": "O2",
    "do_train": true,
    "do_eval": true,
    "disable_tqdm": true,
    "load_best_model_at_end": true,
    "eval_with_do_generation": false,
    "metric_for_best_model": "accuracy",
    "recompute": true,
    "save_total_limit": 1,
    "tensor_parallel_degree": 1,
    "pipeline_parallel_degree": 1,
    "sharding": "stage2",
    "zero_padding": false,
    "unified_checkpoint": true,
    "use_flash_attention": false
  }
```
更多 sft_argument.json 配置文件说明，请参考[大模型精调](../docs/finetune.md)


<a name="模型推理"></a>

#### 4.4 模型推理

通过运行以下命令进行模型评估：
```shell
# 返回到llm目录下
python ./predict/predictor.py \
    --model_name_or_path ./checkpoints/sft_ckpts \
    --dtype float16 \
    --data_file ./ie/data/test.json \
    --output_file ./output/output.json \
    --src_length  512 \
    --max_length  20 \
    --batch_size  4

```


<a name="定制模型一键预测"></a>

#### 4.5 定制模型一键预测

`paddlenlp.Taskflow`装载定制模型，通过`task_path`指定模型权重文件的路径，路径下需要包含训练好的模型权重文件



<a name="模型快速服务化部署"></a>

#### 4.6 模型快速服务化部署
在 UIE 的服务化能力中我们提供基于 PaddleNLP SimpleServing 来搭建服务化能力，通过几行代码即可搭建服务化部署能力



<a name="实验指标"></a>

#### 4.7 实验指标

<!-- 我们在互联网、医疗、金融三大垂类自建测试集上进行了实验：

<table>
<tr><th row_span='2'><th colspan='2'>金融<th colspan='2'>医疗<th colspan='2'>互联网
<tr><td><th>0-shot<th>5-shot<th>0-shot<th>5-shot<th>0-shot<th>5-shot
<tr><td>uie-base (12L768H)<td>46.43<td>70.92<td><b>71.83</b><td>85.72<td>78.33<td>81.86
<tr><td>uie-medium (6L768H)<td>41.11<td>64.53<td>65.40<td>75.72<td>78.32<td>79.68
<tr><td>uie-mini (6L384H)<td>37.04<td>64.65<td>60.50<td>78.36<td>72.09<td>76.38
<tr><td>uie-micro (4L384H)<td>37.53<td>62.11<td>57.04<td>75.92<td>66.00<td>70.22
<tr><td>uie-nano (4L312H)<td>38.94<td>66.83<td>48.29<td>76.74<td>62.86<td>72.35
<tr><td>uie-m-large (24L1024H)<td><b>49.35</b><td><b>74.55</b><td>70.50<td><b>92.66</b><td><b>78.49</b><td><b>83.02</b>
<tr><td>uie-m-base (12L768H)<td>38.46<td>74.31<td>63.37<td>87.32<td>76.27<td>80.13
</table>

0-shot 表示无训练数据直接通过 ```paddlenlp.Taskflow```进行预测，5-shot 表示每个类别包含5条标注数据进行模型微调。**实验表明 UIE 在垂类场景可以通过少量数据（few-shot）进一步提升效果**。 -->


<a name="模型部署"></a>

#### 4.8 模型部署

以下是 UIE Python 端的部署流程，包括环境准备、模型导出和使用示例。

