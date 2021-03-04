import tensorflow as tf


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, dec_hidden, enc_output):
        """
        :param dec_hidden: shape=(16, 300)需要增维
        :param enc_output: shape=(16, 200, 300)
        :param enc_padding_mask: shape=(16, 200)
        :param use_coverage:
        :param prev_coverage: None
        :return:
        """
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(dec_hidden, 1) 
        #print("hidden_with_time_axis",hidden_with_time_axis.get_shape())
        #hidden_with_time_axis (16, 1, 300)
        #第二次进来：hidden_with_time_axis (16, 1, 220)
        #print("W1",self.W1.variables)
        #print("W2",self.W2.variables)
        #print("V",self.V.variables)
        '''
        W1 []
        W2 []
        V []
        第二次进来：
        W1.k.shape=(300,260)有值,W1.b.shape=(260,)=0
        W2.k.shape=(300,260)有值,W2.b.shape=(260,)=0
        W1.k.shape=(260，1),W1.b.shape=(1,)=0
        '''
        
        att_features = self.W1(enc_output) + self.W2(hidden_with_time_axis)

        # Calculate v^T tanh(W_h h_i + W_s s_t + b_attn)
        score = self.V(tf.nn.tanh(att_features))
        #print("score:",score)
        '''
        [[[-0.10661406]
        [-0.14066157]
        [-0.14689432]
        ...
        [-0.0707678 ]
        [-0.05201063]
        [ 0.04659015]]
        ...
        [[]...[]]],shape=(16, 200, 1)=(batch_size,enc_max_len,1)
        
        '''
        """
        定义score
        your code
        """
        # Calculate attention distribution
        
        """
        归一化score，得到attn_dist
        your code
        """
        attn_dist = tf.nn.softmax(score,axis=1)#shape= (16,200,1)
        #print("attn_dist:",attn_dist)
        '''
        [[[0.00435685]
        [0.00421101]
        [0.00418484]
        ...
        [0.00451586]
        [0.00460137]
        [0.00507818]]
        ...
        [[]...[]]],shape=(16, 200, 1)=(batch_size,enc_max_len,1)

        '''
        
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attn_dist * enc_output  # shape=(16, 200, 300)
        #print("context_vector",context_vector)
        '''
        [[[-9.6328651e-05 -6.7108846e-04 -3.4547251e-04 ...  3.1042789e-04    8.3610904e-04 -1.5858321e-04]
        ...
        [-4.1244342e-04  5.0878263e-04 -1.0145822e-03 ... -1.2361992e-04   -5.5835483e-04 -5.6707015e-04]]
        ...
        [[...]
        ...
        [...]]],shape=(16, 200, 300)=(batch_size,enc_max_len,enc_hidden_size)
        '''
        context_vector = tf.reduce_sum(context_vector, axis=1)  # shape=(16, 300)#中间维度上进行求和，结果就是中间维度求和之后没有了
        #print("context_vector2",context_vector)
        '''
        [[-0.01004136 -0.00849795 -0.02748142 ...  0.10412269 -0.02657067  -0.10651132]
        ...
        [-0.03368178 -0.06051109 -0.02596242 ...  0.09218337 -0.06119338  -0.11394408]], shape=(16, 300), dtype=float32)
        '''
        return context_vector, tf.squeeze(attn_dist, -1)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, embedding_matrix):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        """
        定义Embedding层，加载预训练的词向量
        your code
        """
        #定义Embedding层，加载预训练的词向量
        self.embedding = tf.keras.layers.Embedding(vocab_size,
                                                   embedding_dim,
                                                   weights=[embedding_matrix],
                                                   trainable=False)
        
        """
        定义单向的RNN、GRU、LSTM层
        your code
        """
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')


        # self.dropout = tf.keras.layers.Dropout(0.5)
        """
        定义最后的fc层，用于预测词的概率
        your code
        """
        #输出维度是字典大小，激活函数是softmax
        self.fc = tf.keras.layers.Dense(vocab_size,activation=tf.keras.activations.softmax)

    def call(self, x, hidden, enc_output, context_vector):
        # enc_output shape == (batch_size, max_length, hidden_size)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        #x是标签数据，就是摘要的数据
        x = self.embedding(x)
        #print('x is ', x) 
        '''
        x is  tf.Tensor(
        [[[-0.35985306 -0.13879634 -0.09044454 ...  0.16822134 -0.27022406   -0.05214289]]
        [[-0.35985306 -0.13879634 -0.09044454 ...  0.16822134 -0.27022406   -0.05214289]]
        ...
        [[-0.35985306 -0.13879634 -0.09044454 ...  0.16822134 -0.27022406   -0.05214289]]
        [[-0.35985306 -0.13879634 -0.09044454 ...  0.16822134 -0.27022406   -0.05214289]]], shape=(16, 1, 256), dtype=float32),shape==(batch_size, 1, embedding_dim)
        '''
        #context_vector's shape==(batch_size,hidden_size)
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        #两部分信息结合，之后输入到gru中
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        #print("x_concat:",x)
        '''
        x_concat: tf.Tensor(
        [[[ 0.05982509  0.10982166  0.18107766 ...  0.16822134 -0.27022406   -0.05214289]]
        [[ 0.01160153  0.19602422  0.19489945 ...  0.16822134 -0.27022406   -0.05214289]]
        ...
        [[ 0.08859249  0.13527739  0.16346881 ...  0.16822134 -0.27022406   -0.05214289]]
        [[ 0.07347409  0.11560699  0.15718397 ...  0.16822134 -0.27022406   -0.05214289]]], shape=(16, 1, 556), dtype=float32),shape == (batch_size, 1, embedding_dim + enc_hidden_size)
        '''
        # passing the concatenated vector to the GRU
        output, state = self.gru(x)
        #print("output,state:",output,state)
        '''
        output,state: tf.Tensor(
        [[[-8.43024477e-02  9.93552059e-02  1.01860538e-01 ... -1.62877515e-01   -1.12113632e-01 -2.39732815e-03]]
        ...
        [[-8.74022022e-02  1.09095596e-01  9.41279531e-02 ... -1.41359374e-01   -1.11774772e-01  6.39707362e-03]]], shape=(16, 1, 220), dtype=float32),shape==(batch_size, 1, dec_hidden_size)
        
        tf.Tensor(
        [[-8.43024477e-02  9.93552059e-02  1.01860538e-01 ... -1.62877515e-01  -1.12113632e-01 -2.39732815e-03]
        [-1.07611045e-01  9.42999274e-02  1.04836687e-01 ... -1.42188594e-01  -1.07981443e-01 -2.20312993e-03]
        [-1.29673138e-01  8.25620666e-02  1.33741677e-01 ... -1.33564964e-01  -9.21256915e-02 -2.30463617e-03]
        ...
        [-1.14237890e-01  9.45391059e-02  1.15216784e-01 ... -1.55073702e-01  -1.04259707e-01 -3.75023717e-03]
        [-1.07609294e-01  1.04104005e-01  1.06481947e-01 ... -1.55050606e-01  -1.12085924e-01  5.22642549e-05]
        [-8.74022022e-02  1.09095596e-01  9.41279531e-02 ... -1.41359374e-01  -1.11774772e-01  6.39707362e-03]], shape=(16, 220), dtype=float32),shape==(batch_size,dec_hidden_size)
        '''
        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))#(batch_size,hidden_size)
        #print("out_reshape:",output)
        '''
        out_reshape: tf.Tensor(
        [[-8.43024477e-02  9.93552059e-02  1.01860538e-01 ... -1.62877515e-01  -1.12113632e-01 -2.39732815e-03]
        [-1.07611045e-01  9.42999274e-02  1.04836687e-01 ... -1.42188594e-01  -1.07981443e-01 -2.20312993e-03]
        [-1.29673138e-01  8.25620666e-02  1.33741677e-01 ... -1.33564964e-01  -9.21256915e-02 -2.30463617e-03]
        ...
        [-1.14237890e-01  9.45391059e-02  1.15216784e-01 ... -1.55073702e-01  -1.04259707e-01 -3.75023717e-03]
        [-1.07609294e-01  1.04104005e-01  1.06481947e-01 ... -1.55050606e-01  -1.12085924e-01  5.22642549e-05]
        [-8.74022022e-02  1.09095596e-01  9.41279531e-02 ... -1.41359374e-01  -1.11774772e-01  6.39707362e-03]], shape=(16, 220), dtype=float32),shape==(batch_size,dec_hidden_size)
        '''
        
        # output = self.dropout(output)
        #得到概率分布out
        out = self.fc(output)#(batch_size,hidden_size,1)
        #print("out:",out)
        '''
        out: tf.Tensor(
        [[0.00019656 0.00019541 0.00019673 ... 0.00019762 0.00019507 0.00020843]
        [0.00019705 0.00019587 0.0001963  ... 0.00019784 0.00019484 0.00020801]
        [0.0001966  0.00019549 0.00019424 ... 0.00019747 0.0001933  0.00020799]
        ...
        [0.00019682 0.00019553 0.00019642 ... 0.00019737 0.0001952  0.0002085 ]
        [0.00019697 0.00019562 0.00019641 ... 0.00019768 0.00019467 0.00020868]
        [0.00019677 0.00019552 0.00019732 ... 0.00019744 0.00019607 0.00020806]], shape=(16, 5000), dtype=float32),shape==(batch_size,vocab_size)
        '''

        return x, out, state

