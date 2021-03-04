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
        # [batch_sz, max_train_x, enc_units], [batch_sz, enc_units]
        #print("dd",enc_inp, enc_hidden)
        '''
        enc_inp:
        [[ 141    4    0 ...    4  134   23]
        [ 381    0    0 ...    1    1    1]
        [  79    0    0 ...    1    1    1]
        ...
        [  37    5    9 ...    1    1    1]
        [1663  462    0 ...    1    1    1]
        [   0  273  764 ...    7   16    9]],shape=(16, 200), dtype=int32)
        
        enc_hidden:
        [[0. 0. 0. ... 0. 0. 0.]
        ...
        [0. 0. 0. ... 0. 0. 0.]], shape=(16, 200), dtype=float32)
        '''
        enc_output, enc_hidden = self.encoder(enc_inp, enc_hidden)
        #print("enc_out:",enc_output,enc_hidden)
        '''
        enc_output:shape=(batch_size,enc_max_len,enc_units=enc_hidden_size)(一批多少条数据，一条数据多少的字符id上限，每一个字符对应的向量表示)
        [[[-0.07289861 -0.13095766  ...   0.06749522   -0.14931895]
        ...
        [ 0.148442   -0.2581959  ...   -0.1795936    0.00754423]]
        ...
        [[...]
        ...
        [...]]], shape=(16, 200, 200), dtype=float32
        
        
        enc_hidden：shape=(batch_size,enc_units=enc_hidden_size)
         tf.Tensor(
        [[ 0.148442   -0.2581959  -0.09417841 ...  0.01619582  0.06749522  -0.14931895]
        [ 0.12628604  0.15115693  0.27908444 ...  0.17608134  0.0418846   0.14491011]
        [ 0.12628603  0.15115695  0.27908444 ...  0.04697812  0.03159551  -0.19598679]
        ...
        [ 0.12628621  0.151157    0.27908465 ...  0.06738275 -0.01532332  -0.02376204]
        [ 0.1263087   0.1511167   0.27907002 ...  0.10701587  0.06129506   0.02773223]
        [ 0.24870747 -0.14900112 -0.00668985 ...  0.10235632  0.14419761  -0.29494458]], shape=(16, 200), dtype=float32)
        '''
        
        return enc_output, enc_hidden
    
    def call(self, enc_output, dec_inp, dec_hidden, dec_tar):
        predictions = []
        attentions = []
        #初始化Attention，主要是初始化隐藏层和内部各个参数，隐藏层维度先初始化与encoder一致
        context_vector, _ = self.attention(dec_hidden,  # shape=(16, 300)
                                           enc_output) # shape=(16, 200, 300)
        #print("dec_max_len:",dec_tar.shape[1])
        for t in range(dec_tar.shape[1]): # 40（dec_maxlen）
            # Teachering Forcing
            """
            应用decoder来一步一步预测生成词语概论分布
            your code
            如：xxx = self.decoder(), 采用Teachering Forcing方法
            """
            #输入真实结果切片
            x_input = dec_inp[:,t]
            #print("x_input:",x_input)
            #x_input: tf.Tensor([2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2], shape=(16,), dtype=int32)相当于同时取出来了一批batch_size个的同一索引位置的数据进行训练，比如第一次就是取的全是2，开始符STA
            #x_input维度不匹配需要增维
            _, pred, dec_hidden = self.decoder(tf.expand_dims(x_input,1),
                                                dec_hidden,
                                                enc_output,
                                                context_vector)
            #此处的attention就是第二次进Attention的值u，其实就是正常开始训练之后的值，不同于初始化的值
            context_vector, attn_dist = self.attention(dec_hidden, enc_output)
            
            predictions.append(pred)
            #print("predictions",len(predictions),predictions[0].get_shape())
            '''
            predictions [<tf.Tensor: shape=(16, 5000), dtype=float32, numpy=
array([[0.00019526, 0.00020337, 0.00019758, ..., 0.00020023, 0.00019799,0.00020004],
       [0.00019573, 0.00020318, 0.0001963 , ..., 0.00020024, 0.00019801,0.00019947],
       [0.00019485, 0.00020241, 0.00019474, ..., 0.00019971, 0.00019704,0.00019781],
       ...,
       [0.0001954 , 0.00020332, 0.00019719, ..., 0.00020011, 0.00019791,0.0001992 ],
       [0.00019549, 0.00020307, 0.00019725, ..., 0.00020029, 0.00019803,0.00019954],
       [0.00019625, 0.00020275, 0.00019716, ..., 0.00020123, 0.0001987 ,0.00020035]], dtype=float32)>]
            '''
            attentions.append(attn_dist)
            #print("attentions",len(attentions),attentions[0].get_shape())
            '''
            predictions，predictions是一个list,list的每一个元素是一个tensor, list长度从1,2,3..38,39,40,...
            [输出的每个字符位置[batch_size个批次,每一批次数据在该位置的预测输出值]，[40(16,5000)]
            --就是每一条文章预测出来的那个摘要的长度，一个epoch，
            对应batch_size条数据，每一条数据都要计算dec_maxlen次decoder和attention;
            一个批次的16条数据并行计算，一次计算出16个同一位置的概率输出值，下一次再计算下一位置的16个输出值，一个epoch，需要样本总数NUM_SAMPLES//batch_size次；
            这里设置epoch=2,NUM_SAMPLES=150,batch_size=16,则要计算2(150//16)=18次；注意用GPU可以矩阵并行计算的意义便在此；
            每计算一次，都要生成一个pred，pred作为list的一个元素是一个tensor,每一个tensor的shape都是(16, 5000)，就是每一个字符经过gru和fc之后输出的词表大小上的概率值
            attentions是一个list,list的每一个元素是一个tensor,每一个tensor的shape都是(16,200),每一个元素都是对应字符的Attention_weight，也就是score经过归一化之后的那个值
            attentions [<tf.Tensor: shape=(16, 200), dtype=float32, numpy=<tf.Tensor: shape=(16, 200), dtype=float32, numpy=
array([[5.7294030e-05, 1.9326000e-04, 4.6946149e-04, ..., 1.9290155e-03,5.9212360e-04, 6.6959903e-05],
       [1.1094903e-04, 3.3746383e-04, 1.3027202e-03, ..., 1.5229753e-03,3.0046416e-04, 7.6127100e-05],
       [2.8113627e-05, 1.1661458e-04, 5.6261825e-04, ..., 2.1908749e-03,6.8081234e-04, 7.8631529e-05],
       ...,
       [1.6782053e-04, 9.6646562e-04, 2.2823892e-03, ..., 1.9596957e-03,4.4803810e-04, 5.9194685e-05],
       [7.0652604e-05, 3.0334899e-04, 9.8035252e-04, ..., 4.5641307e-03,1.4652200e-03, 1.7887837e-04],
       [8.5009298e-05, 3.1370105e-04, 1.0692772e-03, ..., 9.2025270e-04,2.7673997e-04, 2.1542044e-05]], dtype=float32)>,
            
            '''

        return tf.stack(predictions, 1), dec_hidden