import tensorflow as tf
import time
from seq2seq_pgn_tf2.utils.losses import loss_function
START_DECODING = '[START]'


def train_model(model, dataset, params, ckpt, ckpt_manager):
    # optimizer = tf.keras.optimizers.Adagrad(params['learning_rate'],
    #                                         initial_accumulator_value=params['adagrad_init_acc'],
    #                                         clipnorm=params['max_grad_norm'])
    optimizer = tf.keras.optimizers.Adam(name='Adam', learning_rate=params["learning_rate"])

    # @tf.function()
    def train_step(enc_inp, enc_extended_inp, dec_inp, dec_tar, batch_oov_len, enc_padding_mask, padding_mask):
        # loss = 0
        #enc_inp,dec_tar,dec_inp: (16, 200) (16, 40) (16, 40)
        #print("enc_inp,dec_tar,dec_inp:",enc_inp.get_shape(),dec_tar.get_shape(),dec_inp.get_shape())
        with tf.GradientTape() as tape:
            #初始化encoder状态，一堆0
            enc_output, enc_hidden = model.call_encoder(enc_inp)#(16,200,200),(16,200)
            
            #初始化decoder状态等于encoder状态
            dec_hidden = enc_hidden
            outputs = model(enc_output,  # shape=(3, 200, 256)
                            dec_hidden,  # shape=(3, 256)
                            enc_inp,  # shape=(3, 200)
                            enc_extended_inp,  # shape=(3, 200)
                            dec_inp,  # shape=(3, 50)
                            batch_oov_len,  # shape=()
                            enc_padding_mask,  # shape=(3, 200)
                            params['is_coverage'],
                            prev_coverage=None)
            loss = loss_function(dec_tar,
                                 outputs,
                                 padding_mask,
                                 params["cov_loss_wt"],
                                 params['is_coverage'])
        
        # variables = model.trainable_variables
        variables = model.encoder.trainable_variables +\
                    model.attention.trainable_variables +\
                    model.decoder.trainable_variables +\
                    model.pointer.trainable_variables
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
                              batch[0]["extended_enc_input"],  # shape=(16, 200)
                              batch[1]["dec_input"],  # shape=(16, 50)
                              batch[1]["dec_target"],  # shape=(16, 50)
                              batch[0]["max_oov_len"],  # ()
                              batch[0]["sample_encoder_pad_mask"],  # shape=(16, 200)
                              batch[1]["sample_decoder_pad_mask"])  # shape=(16, 50)

            step += 1
            total_loss += loss
            if step % 100 == 0:
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

