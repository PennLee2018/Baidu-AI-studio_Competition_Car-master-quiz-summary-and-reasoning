import tensorflow as tf

#vocab_size=30000，embedding_dim=256(修改后)，enc_units=200，batch_sz=256，embedding_matrix
class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, embedding_matrix):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        # self.enc_units = enc_units
        self.enc_units = enc_units // 2
        #定义Embedding层，加载预训练的词向量
        #self.embedding_matrix = embedding_matrix
        #初始化embedding层，字典尺寸和embedding维度是两个重要参数,输入vocab_size，输入词向量矩阵（vocab_size，embedding_dim）
        #输出（batch_size,max_lenth,embedding_dim）
        self.embedding = tf.keras.layers.Embedding(vocab_size,
                                                          embedding_dim,
                                                          weights=[embedding_matrix],
                                                          trainable=False)
        #定义单向的RNN、GRU、LSTM层,最重要的参数是输出的维度enc_units，单向是多少维，双向是多少维,自己定义，可以设置和词向量维度一样
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')#权值矩阵初始化方式为正态分布
        # tf.keras.layers.GRU自动匹配cpu、gpu
        #定义双向的gru,采用的方式是拼接
        self.bigru = tf.keras.layers.Bidirectional(self.gru, merge_mode='concat')

    def call(self, x, hidden):
        #x传的是token_id
        #输入是enc_inp:(batch_size,encinp_max_len)
        x = self.embedding(x)
        #print("x_emb.shape=",x,x.get_shape())
        '''
        x.embedding:shape=(batch_size,encinp_max_len,embedding_size)
        [[[-0.3305871  -0.28042006  0.14996733 ... -0.077479    0.46052778    0.00563224]
        [-0.41682342  0.05345337 -0.01910741 ... -0.07651655 -0.10543183   -0.29283813]
        [-0.15476528  0.14495616  0.1286864  ...  0.07453883  0.08910657   -0.13799995]
        ...
        [-0.41682342  0.05345337 -0.01910741 ... -0.07651655 -0.10543183   -0.29283813]
        [-0.13435796 -0.07514819 -0.14962153 ...  0.12196986 -0.18665642   -0.31685084]
        [-0.09824581 -0.5393529  -0.34179434 ...  0.5926068  -0.8939888    0.19892275]]
        ...
        [[...]
        ...
        [...]], shape=(16, 200, 256), dtype=float32)
        
        '''
        #把隐藏层划分成几个子张量，分成2份，在第二个维度上进行切分
        hidden = tf.split(hidden, num_or_size_splits=2, axis=1)
        #print("hidden_split:",hidden)
        '''
        hidden此刻是一个list:[tensor1:shape=(batch_size,enc_hidden_size/2)[[0.,0.,0....,0.,0.],...,[0...]],tensor2 as same as tensor1]
        '''
        
        output, forward_state, backward_state = self.bigru(x, initial_state=hidden)
        #print("bigru:",output.get_shape(),forward_state.get_shape(),backward_state.get_shape())
        '''
        bigru: (16, 200, 300) (16, 150) (16, 150),(batch_size,enc_max_len,enc_units=enc_hidden_size),(batch_size,enc_units/2),(batch_size,enc_units/2)
        '''
        state = tf.concat([forward_state, backward_state], axis=1)
        # output, state = self.gru(x, initial_state=hidden)
        return output, state #输出encoder的输出和隐藏层状态
    
    
    #初始化第一个encoder全0
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, 2*self.enc_units))
    
