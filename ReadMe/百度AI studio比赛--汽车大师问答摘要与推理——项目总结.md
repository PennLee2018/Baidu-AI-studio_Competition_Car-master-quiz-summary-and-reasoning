# 百度AI studio比赛--汽车大师问答摘要与推理——项目总结

## 项目背景描述

项目是由百度AI技术生态部门提供，题目为“汽车大师问答摘要与推理”。

要求大家使用汽车大师提供的11万条（技师与用户的多轮对话与诊断建议报告数据）建立模型，模型需基于对话文本、用户问题、车型与车系，输出包含摘要与推断的报告文本，综合考验模型的归纳总结与推断能力。该解决方案可以节省大量人工时间，提高用户获取回答和解决方案的效率。

## 数据集说明

对于每个用户问题"QID"，有对应文本形式的文本集合 D = "Brand", "Collection", "Problem", "Conversation"，要求阅读理解系统自动对D进行分析，输出相应的报告文本"Report"，其中包含摘要与推理。目标是"Report"可以正确、完整、简洁、清晰、连贯地对D中的信息作归纳总结与推理。

训练：所提供的训练集（82943条记录）建立模型，基于汽车品牌、车系、问题内容与问答对话的文本，输出建议报告文本

输出结果：对所提供的测试集（20000条记录）使用训练好的模型，输出建议报告的结果文件，通过最终测评得到评价分数。

### 数据集概览

```python
import pandas as pd
train_path = "E:/课程/07 名企班必修课1-导师制名企实训班自然语言处理方向 004期/Lesson1-项目导论与中文词向量实践/数据集/AutoMaster_TrainSet.csv"
test_path = "E:/课程/07 名企班必修课1-导师制名企实训班自然语言处理方向 004期/Lesson1-项目导论与中文词向量实践/数据集/AutoMaster_TrainSet.csv"
df = pd.read_csv(train_path, encoding='utf-8')
df.head()
```

原始数据展示：

![image-20210304155927892](http://pennlee-aliyun.oss-cn-beijing.aliyuncs.com/img/image-20210304155927892.png)

## 大体流程

> 超参数定义->设置使用GPU->训练->验证->测试

### 超参数定义

> 代码详见/bin/main.py

关键参数简要说明：

```python
max_enc_len = 200 #encoder输入最大长度
max_dec_len = 40 #decoder输入最大长度
batch_size = 32
beam_size = 3
vocab_size = 30000
embed_size = 256
enc_units = 256
dec_units = 256
attn_units = 256
learning_rate = 0.001
steps_per_epoch = 1300
max_steps = 10000
epochs = 20
dropout_rate = 0.1
```

### 设置使用GPU

```python
#设置使用GPU还是CPU
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')

#若使用GPU，对使用哪一块GPU进行设置
if gpus:
    tf.config.experimental.set_visible_devices(devices=gpus[0],device_type='GPU')
```

### 数据预处理

> 数据读取与构建词向量，代码详见preprocess.py、data_reader.py、build_w2v.py。

![image-20210304160948545](http://pennlee-aliyun.oss-cn-beijing.aliyuncs.com/img/image-20210304160948545.png)

> 数据分批与数据规范化

![image-20210304161926269](http://pennlee-aliyun.oss-cn-beijing.aliyuncs.com/img/image-20210304161926269.png)

### 模型构建-Seq2Seq

定义Encoder与Decoder模块，及中间的Attention部分。

Encoder部分：

> *embedding*维度256维
>
> ```python
> self.embedding = tf.keras.layers.Embedding(vocab_size,                                                      					embedding_dim,
>                                              weights=[embedding_matrix],
>                                              trainable=False)
> ```
>
> 网络采用双向GRU作为编码器（由两个单向的GRU拼接而来）。
>
> ```python
> self.gru = tf.keras.layers.GRU(self.enc_units,
>                                        return_sequences=True,
>                                        return_state=True,
>                                        recurrent_initializer='glorot_uniform')
> self.bigru = tf.keras.layers.Bidirectional(self.gru, merge_mode='concat')
> ```
>
> 同时隐藏态也需要拼接：
>
> ```python
> #把隐藏层划分成几个子张量，分成2份，在第二个维度上进行切分
> hidden = tf.split(hidden, num_or_size_splits=2, axis=1)      
> output, forward_state, backward_state = self.bigru(x, initial_state=hidden)
> state = tf.concat([forward_state, backward_state], axis=1)
> ```
>
> 详细代码见rnn_encoder.py。

Attention部分：

> Score计算采用感知机的方式，再将Score经过一个Softmax归一化得到Attention_Weight。
>
> ```python
> # Calculate v^T tanh(W_h h_i + W_s s_t + b_attn)
> score = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis)))
> attn_dist = tf.nn.softmax(score,axis=1)#shape= (16,200,1)
> ```
>
> 然后reduce_sum得到context_vector
>
> ```python
> context_vector = tf.reduce_sum(attn_dist * enc_output, axis=1)
> ```
>
> 详细代码见rnn_decoder.py。

Decoder部分：

> *embedding*维度256维：
>
> ```python
> #定义Embedding层，加载预训练的词向量
> self.embedding = tf.keras.layers.Embedding(vocab_size,
>                                            embedding_dim,
>                                            weights=[embedding_matrix],
>                                            trainable=False)
> ```
>
> 网络采用单向GRU作为编码器:
>
> ```python
> self.gru = tf.keras.layers.GRU(self.dec_units,
>                                return_sequences=True,
>                                return_state=True,
>                                recurrent_initializer='glorot_uniform')
> ```
>
> GRU的输入是由输入x和context_vector组成：
>
> ```
> x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
> ```
>
> GRU输出后需要维度变化然后再过一个FC层输出概率分布：
>
> ```python
> output, state = self.gru(x)
> output = tf.reshape(output, (-1, output.shape[2]))#(batch_size,hidden_size)
> out = self.fc(output)#(batch_size,hidden_size)
> ```
>
> 代码在rnn_encoder.py中。

整体模型定义上还会有一些细节，例如Tearcher的使用等等，整体模型定义代码如下：

```python
import tensorflow as tf
from seq2seq_tf2.encoders import rnn_encoder
from seq2seq_tf2.decoders import rnn_decoder
from utils.data_utils import load_word2vec
import time


class SequenceToSequence(tf.keras.Model):
    def __init__(self, params):
        super(SequenceToSequence, self).__init__()
        #读取word2vec.txt、vocab.txt返回词向量矩阵，矩阵大小vacab_size*embed_size,每一行都是行索引编号对应word的词向量
        self.embedding_matrix = load_word2vec(params)
        self.params = params
        self.encoder = rnn_encoder.Encoder(params["vocab_size"],
                                           params["embed_size"],
                                           params["enc_units"],
                                           params["batch_size"],
                                           self.embedding_matrix)
        self.attention = rnn_decoder.BahdanauAttention(params["attn_units"])
        self.decoder = rnn_decoder.Decoder(params["vocab_size"],
                                           params["embed_size"],
                                           params["dec_units"],
                                           params["batch_size"],
                                           self.embedding_matrix)

    def call_encoder(self, enc_inp):
        enc_hidden = self.encoder.initialize_hidden_state()#[batch_size,enc_units]
        enc_output, enc_hidden = self.encoder(enc_inp, enc_hidden)       
        return enc_output, enc_hidden
    
    def call(self, enc_output, dec_inp, dec_hidden, dec_tar):
        predictions = []
        attentions = []
        #初始化Attention，主要是初始化隐藏层和内部各个参数，隐藏层维度先初始化与encoder一致
        context_vector, _ = self.attention(dec_hidden,  # shape=(16, 300)
                                           enc_output) # shape=(16, 200, 300)
        for t in range(dec_tar.shape[1]): # 40（dec_maxlen）
            # Teachering Forcing
            x_input = dec_inp[:,t]
            _, pred, dec_hidden = self.decoder(tf.expand_dims(x_input,1),
                                                dec_hidden,
                                                enc_output,
                                                context_vector)
            #此处的attention就是第二次进Attention的值u，其实就是正常开始训练之后的值，不同于初始化的值
            context_vector, attn_dist = self.attention(dec_hidden, enc_output)
            predictions.append(pred)
            attentions.append(attn_dist)

        return tf.stack(predictions, 1), dec_hidden
```

### 模型训练

优化器选用Adam:

```python
optimizer = tf.keras.optimizers.Adam(name='Adam', learning_rate=params["learning_rate"])
```

损失函数为交叉熵函数：

```python
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')
# 定义损失函数
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 1))
    dec_lens = tf.reduce_sum(tf.cast(mask, dtype=tf.float32), axis=-1)

    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    loss_ = tf.reduce_sum(loss_, axis=-1)/dec_lens
    return tf.reduce_mean(loss_)
```

定义训练的每一步：

```python
#dec_tar是标签值，dec_inp是正常输入值
def train_step(enc_inp, dec_tar, dec_inp):
    with tf.GradientTape() as tape:
        #初始化encoder状态，一堆0
        enc_output, enc_hidden = model.call_encoder(enc_inp)

        #初始化decoder状态等于encoder状态
        dec_hidden = enc_hidden

        # start index
        #得到50步堆接起来的概率分布值，传入loss计算函数
        #将encoder后的输出以及dec的输入和初始状态的隐藏层状态传入seq2seq模型，得到预测值
        pred, _ = model(enc_output,  # shape=(16, 200, 300)
                        dec_inp,  # shape=(16, 40)
                        dec_hidden,  # shape=(16, 300)
                        dec_tar)  # shape=(16, 40) 
        loss = loss_function(dec_tar, pred)
        #反向梯度更新
        variables = model.encoder.trainable_variables + model.attention.trainable_variables + model.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return loss
```

训练并保存中间结果：

```python
    best_loss = 20
    epochs = params['epochs']
    for epoch in range(epochs):
        t0 = time.time()
        step = 0
        total_loss = 0
        #分批次训练
        for batch in dataset:
            loss = train_step(batch[0]["enc_input"], 
                              batch[1]["dec_target"], 
                              batch[1]["dec_input"])
            step += 1
            total_loss += loss
            if step % 100 == 0:#100
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, step, total_loss / step))
        #存模型数据，每一步存一下
        if epoch % 1 == 0: 
            if total_loss / step < best_loss:
                best_loss = total_loss / step
                ckpt_save_path = ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {} ,best loss {}'.format(epoch + 1, ckpt_save_path, best_loss))
                print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / step))
                print('Time taken for 1 epoch {} sec\n'.format(time.time() - t0))

```

> 相关代码主要见train_helper.py和train_eval_test.py。

### 测试

测试的主要流程与训练相似，可以选择greedy_decode和beam_decode两种方式：

```python
def test(params):
    assert params["mode"].lower() == "test", "change training mode to 'test' or 'eval'"
    # assert params["beam_size"] == params["batch_size"], "Beam size must be equal to batch_size, change the params"

    print("Building the model ...")
    model = SequenceToSequence(params)

    print("Creating the vocab ...")
    vocab = Vocab(params["vocab_path"], params["vocab_size"])

    print("Creating the batcher ...")
    b = batcher(vocab, params)

    print("Creating the checkpoint manager")
    checkpoint_dir = "{}/checkpoint".format(params["seq2seq_model_dir"])
    ckpt = tf.train.Checkpoint(SequenceToSequence=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)

    # path = params["model_path"] if params["model_path"] else ckpt_manager.latest_checkpoint
    # path = ckpt_manager.latest_checkpoint
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Model restored")
    # for batch in b:
    #     yield batch_greedy_decode(model, batch, vocab, params)
    #if params['greedy_decode']:
        # params['batch_size'] = 512
    predict_result(model, params, vocab, params['test_save_dir'])

def predict_result(model, params, vocab, result_save_path):
    dataset = batcher(vocab, params)
    # 预测结果
    if params['greedy_decode']:
        results = greedy_decode(model, dataset, vocab, params)
    elif params['beam_search_decode']:
        results = beam_decode(model, dataset, vocab, params)
    results = list(map(lambda x: x.replace(" ",""), results))
    #print("results2",results)#64个元素，每个元素都是大小为dec_max_len的序列
    # 保存结果
    save_predict_result(results, params)

    return results
```

具体实现方法可见test_helper.py。

### 验证

评价指标Rouge-L

```python
def evaluate(params):
    gen = test(params)
    reals = []
    preds = []
    with tqdm(total=params["max_num_to_eval"], position=0, leave=True) as pbar:
        for i in range(params["max_num_to_eval"]):
            trial = next(gen)
            reals.append(trial.real_abstract)
            preds.append(trial.abstract)
            pbar.update(1)
    r = Rouge()
    scores = r.get_scores(preds, reals, avg=True)
    print("\n\n")
    pprint.pprint(scores)
```

## 其他优化

- 使用PGN网络解决OOV问题
- 采用惩罚机制解决重复词语的问题

## 最终效果

![result](http://pennlee-aliyun.oss-cn-beijing.aliyuncs.com/img/result.png)