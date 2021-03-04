import tensorflow as tf
import time

START_DECODING = '[START]'


def train_model(model, dataset, params, ckpt, ckpt_manager):
    # optimizer = tf.keras.optimizers.Adagrad(params['learning_rate'],
    #                                         initial_accumulator_value=params['adagrad_init_acc'],
    #                                         clipnorm=params['max_grad_norm'])
    optimizer = tf.keras.optimizers.Adam(name='Adam', learning_rate=params["learning_rate"])
    # from_logits = True: preds is model output before passing it into softmax (so we pass it into softmax)
    # from_logits = False: preds is model output after passing it into softmax (so we skip this step)
    # 要看模型在decoder最后输出是否经过softmax
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

    # @tf.function()
    #dec_tar是标签值，dec_inp是正常输入值
    def train_step(enc_inp, dec_tar, dec_inp):
        # loss = 0
        #enc_inp,dec_tar,dec_inp: (16, 200) (16, 40) (16, 40)
        #print("enc_inp,dec_tar,dec_inp:",enc_inp.get_shape(),dec_tar.get_shape(),dec_inp.get_shape())
        with tf.GradientTape() as tape:
            #初始化encoder状态，一堆0
            enc_output, enc_hidden = model.call_encoder(enc_inp)#(16,200,200),(16,200)
            
            #初始化decoder状态等于encoder状态
            dec_hidden = enc_hidden
            #print("dec_hidden:",dec_hidden),shape=(16, 300)
            # start index
            #得到50步堆接起来的概率分布值，传入loss计算函数
            #将encoder后的输出以及dec的输入和初始状态的隐藏层状态传入seq2seq模型，得到预测值
            pred, _ = model(enc_output,  # shape=(16, 200, 300)
                            dec_inp,  # shape=(16, 40)
                            dec_hidden,  # shape=(16, 300)
                            dec_tar)  # shape=(16, 40) 
            loss = loss_function(dec_tar, pred)
                        
        # variables = model.trainable_variables
        #反向梯度更新
        variables = model.encoder.trainable_variables + model.attention.trainable_variables + model.decoder.trainable_variables
        #print("variables",len(variables),variables[0].get_shape)#len=epoch(sample_num//batch_size)-1,shape(256, 384)
        gradients = tape.gradient(loss, variables)
        #print("gradients",len(gradients),gradients[0].get_shape)#len=epoch(sample_num//batch_size)-1,shape(256, 128+128+128）
        optimizer.apply_gradients(zip(gradients, variables))
        return loss

    best_loss = 20
    epochs = params['epochs']
    for epoch in range(epochs):
        t0 = time.time()
        step = 0
        total_loss = 0
        # for step, batch in enumerate(dataset.take(params['steps_per_epoch'])):
        #分批次训练
        for batch in dataset:
        # for batch in dataset.take(params['steps_per_epoch']):
            #==train==:Tensor("IteratorGetNext:2", shape=(16, ?), dtype=int32)==Tensor("IteratorGetNext:10", shape=(16, 40), dtype=int32)==Tensor("IteratorGetNext:8", shape=(16, 40), dtype=int32)
            #print("==train==:{}=={}=={}".format(batch[0]["enc_input"],batch[1]["dec_target"],batch[1]["dec_input"]))
            '''
            一个batch的enc_input：(batch_size,encoder_max_len)
            [[ 141    4    0 ...    4  134   23]
            [ 381    0    0 ...    1    1    1]
            [  79    0    0 ...    1    1    1]
            ...
            [  37    5    9 ...    1    1    1]
            [1663  462    0 ...    1    1    1]
            [   0  273  764 ...    7   16    9]]
            
             一个batch的dec_target：(batch_size,decoder_max_len)
            [[  15    4  278  ...  163  131]
            ...
            [ 172 1171  370  ...   29  8 1177]]
            
            一个batch的dec_input：(batch_size,decoder_max_len),加头或补尾
            [[   2   15    4  278  ...   63  163]
            ...
            [   2  172 1171  370  ... 29    8]]
            '''
            
            loss = train_step(batch[0]["enc_input"],  # shape=(16, 200)
                              batch[1]["dec_target"], # shape=(16, 40)
                              batch[1]["dec_input"])#shape=(16,40)
           
            step += 1
            total_loss += loss
            if step % 100 == 0:#100
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, step, total_loss / step))
                # print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, step, loss.numpy()))
        #存模型数据，每一步存一下
        if epoch % 1 == 0: 
            if total_loss / step < best_loss:
                best_loss = total_loss / step
                ckpt_save_path = ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {} ,best loss {}'.format(epoch + 1, ckpt_save_path, best_loss))
                print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / step))
                print('Time taken for 1 epoch {} sec\n'.format(time.time() - t0))

