import tensorflow as tf

SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]'
UNKNOWN_TOKEN = '[UNK]'
START_DECODING = '[START]'
STOP_DECODING = '[STOP]'

#vocab_file=》vocab.txt,max_size=30000
#建立词表类，与之前vocab.txt的区别在于，加入对特殊符号的处理
class Vocab:
    def __init__(self, vocab_file, max_size):
        self.word2id = {UNKNOWN_TOKEN: 0, PAD_TOKEN: 1, START_DECODING: 2, STOP_DECODING: 3}
        self.id2word = {0: UNKNOWN_TOKEN, 1: PAD_TOKEN, 2: START_DECODING, 3: STOP_DECODING}
        self.count = 4

        with open(vocab_file, 'r', encoding='utf-8') as f:
            #读取词表的每一行，应该的格式形如“说	0”
            for line in f:
                pieces = line.split()
                if len(pieces) != 2:
                    #跳过不合法的数据
                    print('Warning : incorrectly formatted line in vocabulary file : %s\n' % line)
                    continue

                w = pieces[0]#取单词,出现非预期词时报错
                if w in [SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
                    raise Exception(r'<s>, </s>, [UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, '
                                    r'but %s is' % w)
                #出现重复词时报错
                if w in self.word2id:
                    raise Exception('Duplicated word in vocabulary file: %s' % w)

                #建立双向词表，字典形式
                self.word2id[w] = self.count
                self.id2word[self.count] = w
                self.count += 1
                
                #超过最大值时报错退出
                if max_size != 0 and self.count >= max_size:
                    print("max_size of vocab was specified as %i; we now have %i words. Stopping reading."
                          % (max_size, self.count))
                    break

        print("Finished constructing vocabulary of %i total words. Last word added: %s" %
              (self.count, self.id2word[self.count - 1]))
    
    #根据词查对应id，遇到OOV返回UNK：0
    def word_to_id(self, word):
        if word not in self.word2id:
            return self.word2id[UNKNOWN_TOKEN]
        return self.word2id[word]

    #根据id查词，id不合法时报错
    def id_to_word(self, word_id):
        if word_id not in self.id2word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self.id2word[word_id]

    #词表实际大小
    def size(self):
        return self.count

#将文章中的OOV词合集oovs、扩充后的词表大小ids
def article_to_ids(article_words, vocab):
    ids = []
    oovs = []
    unk_id = vocab.word_to_id(UNKNOWN_TOKEN)
    for w in article_words:
        i = vocab.word_to_id(w)
        if i == unk_id:  # If w is OOV
            if w not in oovs:  # Add to list of OOVs
                oovs.append(w)
            oov_num = oovs.index(w)  # This is 0 for the first article OOV, 1 for the second article OOV...
            ids.append(vocab.size() + oov_num)  # This is e.g. 50000 for the first article OOV, 50001 for the second...
        else:
            ids.append(i)
    return ids, oovs

#找到摘要中的OOVs,核对其是不是文章OOVs中的一员，若不是，继续添加
def abstract_to_ids(abstract_words, vocab, article_oovs):
    ids = []
    unk_id = vocab.word_to_id(UNKNOWN_TOKEN)
    for w in abstract_words:
        i = vocab.word_to_id(w)
        if i == unk_id:  # If w is an OOV word
            if w in article_oovs:  # If w is an in-article OOV
                vocab_idx = vocab.size() + article_oovs.index(w)  # Map to its temporary article OOV number
                ids.append(vocab_idx)
            else:  # If w is an out-of-article OOV
                ids.append(unk_id)  # Map to the UNK token id
        else:
            ids.append(i)
    return ids

#未看，暂无引用
def output_to_words(id_list, vocab, article_oovs):
    words = []
    for i in id_list:
        try:
            w = vocab.id_to_word(i)  # might be [UNK]
        except ValueError as e:  # w is OOV
            assert article_oovs is not None, "Error: model produced a word ID that isn't in the vocabulary. " \
                                             "This should not happen in baseline (no pointer-generator) mode"
            article_oov_idx = i - vocab.size()
            try:
                w = article_oovs[article_oov_idx]
            except ValueError as e:  # i doesn't correspond to an article oov
                raise ValueError('Error: model produced word ID %i which corresponds to article OOV %i but this '
                                 'example only has %i article OOVs' % (i, article_oov_idx, len(article_oovs)))
        words.append(w)
    return words

#暂无引用
def abstract_to_sents(abstract):
    """
    Splits abstract text from datafile into list of sentences.
    Args:
    abstract: string containing <s> and </s> tags for starts and ends of sentences
    Returns:
    sents: List of sentence strings (no tags)
    """
    cur = 0
    sents = []
    while True:
        try:
            start_p = abstract.index(SENTENCE_START, cur)
            end_p = abstract.index(SENTENCE_END, start_p + 1)
            cur = end_p + len(SENTENCE_END)
            sents.append(abstract[start_p + len(SENTENCE_START): end_p])
        except ValueError as e:  # no more sentences
            return sents


def get_dec_inp_targ_seqs(sequence, max_len, start_id, stop_id):
    """
    Given the reference summary as a sequence of tokens, return the input sequence for the decoder,
    and the target sequence which we will use to calculate loss. The sequence will be truncated if it is longer
    than max_len. The input sequence must start with the start_id and the target sequence must end with the stop_id
    (but not if it's been truncated).
    Args:
      sequence: List of ids (integers)
      max_len: integer
      start_id: integer
      stop_id: integer
    Returns:
      inp: sequence length <=max_len starting with start_id
      target: sequence same length as input, ending with stop_id only if there was no truncation
    """
    inp = [start_id] + sequence[:]
    target = sequence[:]
    #如果输入长度超过限定最大长度max，inp取start_id+前(max-1)个sequence，target取前max个；否则target加上stop_id,
    if len(inp) > max_len:  # truncate
        inp = inp[:max_len]
        target = target[:max_len]  # no end_token
    else:  # no truncation
        target.append(stop_id)  # end token
    #断言，如果实际情况不满足后面条件，则退出返回错误
    assert len(inp) == len(target)#不管是一个加STA还是另一个加END，最终应该长度都相等，要不是max,要不是len+1
    return inp, target


def example_generator(vocab, train_x_path, train_y_path, test_x_path, max_enc_len, max_dec_len, mode, batch_size):
    #训练数据处理
    if mode == "train":
        #提供文件名自动构造一个dataset/
        dataset_train_x = tf.data.TextLineDataset(train_x_path)
        dataset_train_y = tf.data.TextLineDataset(train_y_path)
        #通过给定的数据集压缩构造一个数据集，形如[(x1,y1),(x2,y2),(x3,y3)]
        train_dataset = tf.data.Dataset.zip((dataset_train_x, dataset_train_y))
        # train_dataset = train_dataset.shuffle(1000, reshuffle_each_iteration=True).repeat()
        # i = 0
        #print("gen",train_dataset)
        for raw_record in train_dataset:
            #编码转换
            article = raw_record[0].numpy().decode("utf-8")
            #print("article",article)
            #article 新车 ， 全款 ， 买 了 半个 月 ， 去 4S店 贴膜 时才 发现 右侧 尾灯 下  (...)。 车主 说 ： 恩

            abstract = raw_record[1].numpy().decode("utf-8")
            #print("abstract",abstract)
            #abstract 你好 ， 像 这种 情况 确实 不好 说 ， 你 如果 不 放心 的话 ， 直接 开去 外面 比较 专业 的 修理厂 ，(...)  四 s 店 进行 重新 喷漆 就 可以 。

            #定义起始符
            start_decoding = vocab.word_to_id(START_DECODING)
            stop_decoding = vocab.word_to_id(STOP_DECODING)
            #print("sta,end",start_decoding,stop_decoding)
            #sta,end 2 3
            ##########################对文章进行处理-enc_input#####################
            #max_enc_len=200,只取前200个进行数据规范化
            article_words = article.split()[:max_enc_len]
            enc_len = len(article_words)
            #print("article_words:",article_words,enc_len)
            #article_words: ['新车', '，', '全款', '，', '买', '了', '半个',(...), '说', '：', '正常', '，', '没', '问题'] 200

            # 添加mark标记，sample_encoder_pad_mask里边全部填1
            sample_encoder_pad_mask = [1 for _ in range(enc_len)]
            #把每一条数据里的各个词先转化为id
            enc_input = [vocab.word_to_id(w) for w in article_words]
            #print("enc_input",enc_input)#在原有词表基础上索引全部后移4，因为前面加了四个特殊字符，例如“新车”一词，原先是137，现在是141
            #enc_input [141, 4, 0, 4, 295, 11, 1996, 1995, 4, 57, 205, 558,(...) 0, 4, 46, 43, 10, 6, 7, 46, 4, 134, 23]

            #print("sample_encoder_pad_mask：",sample_encoder_pad_mask)
            #sample_encoder_pad_mask： [1, 1, 1, 1, 1,...,1]len=200

            #获取文章中的oovs,和加入oovs之后的扩充词表enc_input_extend_vocab
            enc_input_extend_vocab, article_oovs = article_to_ids(article_words, vocab)
            #print("enc_input_extend_vocab",enc_input_extend_vocab,article_oovs)#原先是0的UNK全部在添加OOV之后有了值
            #enc_input_extend_vocab [141, 4, 2054, 4, 295, 11, (...), 6, 7, 46, 4, 134, 23] ['全款', '时才', '右侧', '尾灯', '间', '一小', '点掉', '修补', '常见', '补漆笔', '恩', '很快', '℃']

            ##########################对摘要进行处理-dec_inp##########################
            #基本流程是将上面的一条文本拿过来之后先变成word_list形式，再根据vocab变成word_id_list形式,再处理OOV问题
            abstract_sentences = [""]
            #摘要分词
            abstract_words = abstract.split()
            #print("abstract_words",abstract_words)
            #abstract_words ['你好', '，', '像', '这种', '情况', '确实', '不好', '说', (...)四', 's', '店', '进行', '重新', '喷漆', '就', '可以', '。']

            #一条摘要里的词都转为id
            abs_ids = [vocab.word_to_id(w) for w in abstract_words]
            #print("abs_ids",abs_ids)
            #abs_ids [15, 4, 278, 35, 22, 235, 167, 6, 4, 18, 26, 28, 488, 39,(...) ,162, 370, 195, 260, 196, 63, 163, 131, 20, 12, 8]

            abs_ids_extend_vocab = abstract_to_ids(abstract_words, vocab, article_oovs)
            #摘要的词数限定为max_dec_len=40,不足时取实际的长度，dec_input可以作为后面teacherforcing的输入，因为带着STA；target可以作为解码阶段的输出label,可能带着END
            dec_input, target = get_dec_inp_targ_seqs(abs_ids, max_dec_len, start_decoding, stop_decoding)
            _, target = get_dec_inp_targ_seqs(abs_ids_extend_vocab, max_dec_len, start_decoding, stop_decoding)

            dec_len = len(dec_input)
            #print("dec_inp:",dec_input,dec_len)
            #注意与enc_inp不一样的地方，开头加了START_DECODING：2，或结尾加上STOP_DECODING: 3，依据长度来
            #dec_inp: [2, 15, 4, 278, 35, 22, 235, (...), 36, 4, 26, 199, 195, 260, 196, 63, 163] 40

            #print("target",target)#没有加START_DECODING的真实摘要
            #target [15, 4, 278, 35, 22, 235, (...), 36, 4, 26, 199, 195, 260, 196, 63, 163, 131]

            # 添加mark标记
            sample_decoder_pad_mask = [1 for _ in range(dec_len)]
            #print("sample_decoder_pad_mask",sample_decoder_pad_mask)
            #sample_decoder_pad_mask [1, 1, 1, 1, (...), 1, 1, 1, 1, 1, 1] 40

            '''
            article 新车 ， 全款 ， 买 了 半个 月 ， 去 4S店 贴膜 时才 发现 右侧 尾灯 下 缝隙 间 有 一小 点掉 漆 ， 怎么办 ？ 4S店 人员 说 很 正常 。 要 不 就 修补 一下 ， 要 不 自己 买 一点 修补 包 ， 这种 情况 怎么办 技师 说 ： 你好 ， 像 这种 情况 确实 不好 说 ， 你 如果 不 放心 的话 ， 直接 开去 外面 比较 专业 的 修理厂 ， 让 专业 的 喷漆 师傅 帮 你 检查一下 ， 如果 确定 四 s 店 进行 重新 喷漆 的话 ， 这种 情况 属于 四 ｓ 存在 消费 欺诈 ， 可以 走 法律 程序 退一 赔 三 如果 感觉 问题 不 大 的话 ， 直接 要求 四 s 店 进行 重新 喷漆 就 可以 。 车主 说 ： 这种 情况 也 不 常见 技师 说 ： 不 常见 车主 说 ： 也 没有 什么 好 的 办法 技师 说 ： 重新 补漆 ， 或者 是 用 补漆笔 涂抹 一下 。 没有 特别 好 的 办法 车主 说 ： 恩 ， 谢谢 车主 说 ： 你好 车主 说 ： 车 发动 起来 ， 水温 很快 到 90 ℃ ， 正常 吗 技师 说 ： 正常 ， 没 问题 你 可以 关注 我 ， 如果 别的 问题 ， 可以 一对一 咨询 我 。 车主 说 ： 好 的 车主 说 ： 怎么 判断 4s店 处理 过漆面 技师 说 ： 这个 没有 一定 专业 的 知识 ， 这个 是 无法 识别 的 。 车主 说 ： 恩
            abstract 你好 ， 像 这种 情况 确实 不好 说 ， 你 如果 不 放心 的话 ， 直接 开去 外面 比较 专业 的 修理厂 ， 让 专业 的 喷漆 师傅 帮 你 检查一下 ， 如果 确定 四 s 店 进行 重新 喷漆 的话 ， 这种 情况 属于 四 ｓ 存在 消费 欺诈 ， 可以 走 法律 程序 退一 赔 三 如果 感觉 问题 不 大 的话 ， 直接 要求 四 s 店 进行 重新 喷漆 就 可以 。
            sta,end 2 3
            article_words: ['新车', '，', '全款', '，', '买', '了', '半个', '月', '，', '去', '4S店', '贴膜', '时才', '发现', '右侧', '尾灯', '下', '缝隙', '间', '有', '一小', '点掉', '漆', '，', '怎么办', '？', '4S店', '人员', '说', '很', '正常', '。', '要', '不', '就', '修补', '一下', '，', '要', '不', '自己', '买', '一点', '修补', '包', '，', '这种', '情况', '怎么办', '技师', '说', '：', '你好', '，', '像', '这种', '情况', '确实', '不好', '说', '，', '你', '如果', '不', '放心', '的话', '，', '直接', '开去', '外面', '比较', '专业', '的', '修理厂', '，', '让', '专业', '的', '喷漆', '师傅', '帮', '你', '检查一下', '，', '如果', '确定', '四', 's', '店', '进行', '重新', '喷漆', '的话', '，', '这种', '情况', '属于', '四', 'ｓ', '存在', '消费', '欺诈', '，', '可以', '走', '法律', '程序', '退一', '赔', '三', '如果', '感觉', '问题', '不', '大', '的话', '，', '直接', '要求', '四', 's', '店', '进行', '重新', '喷漆', '就', '可以', '。', '车主', '说', '：', '这种', '情况', '也', '不', '常见', '技师', '说', '：', '不', '常见', '车主', '说', '：', '也', '没有', '什么', '好', '的', '办法', '技师', '说', '：', '重新', '补漆', '，', '或者', '是', '用', '补漆笔', '涂抹', '一下', '。', '没有', '特别', '好', '的', '办法', '车主', '说', '：', '恩', '，', '谢谢', '车主', '说', '：', '你好', '车主', '说', '：', '车', '发动', '起来', '，', '水温', '很快', '到', '90', '℃', '，', '正常', '吗', '技师', '说', '：', '正常', '，', '没', '问题'] 200
            enc_input [141, 4, 0, 4, 295, 11, 1996, 1995, 4, 57, 205, 558, 0, 399, 0, 0, 98, 1367, 0, 14, 0, 0, 500, 4, 393, 17, 205, 1025, 6, 120, 46, 8, 74, 28, 20, 0, 33, 4, 74, 28, 583, 295, 188, 0, 1007, 4, 35, 22, 393, 10, 6, 7, 15, 4, 278, 35, 22, 235, 167, 6, 4, 18, 26, 28, 488, 39, 4, 162, 1233, 677, 51, 489, 5, 106, 4, 353, 489, 5, 131, 182, 254, 18, 36, 4, 26, 199, 195, 260, 196, 63, 163, 131, 39, 4, 35, 22, 261, 195, 1234, 222, 1235, 1236, 4, 12, 200, 1237, 478, 1238, 1239, 426, 26, 126, 23, 28, 81, 39, 4, 162, 370, 195, 260, 196, 63, 163, 131, 20, 12, 8, 13, 6, 7, 35, 22, 66, 28, 0, 10, 6, 7, 28, 0, 13, 6, 7, 66, 25, 50, 31, 5, 223, 10, 6, 7, 163, 1767, 4, 65, 9, 60, 0, 672, 33, 8, 25, 382, 31, 5, 223, 13, 6, 7, 0, 4, 83, 13, 6, 7, 15, 13, 6, 7, 47, 1183, 978, 4, 360, 0, 62, 1319, 0, 4, 46, 43, 10, 6, 7, 46, 4, 134, 23]
            sample_encoder_pad_mask： [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            enc_input_extend_vocab [141, 4, 2054, 4, 295, 11, 1996, 1995, 4, 57, 205, 558, 2055, 399, 2056, 2057, 98, 1367, 2058, 14, 2059, 2060, 500, 4, 393, 17, 205, 1025, 6, 120, 46, 8, 74, 28, 20, 2061, 33, 4, 74, 28, 583, 295, 188, 2061, 1007, 4, 35, 22, 393, 10, 6, 7, 15, 4, 278, 35, 22, 235, 167, 6, 4, 18, 26, 28, 488, 39, 4, 162, 1233, 677, 51, 489, 5, 106, 4, 353, 489, 5, 131, 182, 254, 18, 36, 4, 26, 199, 195, 260, 196, 63, 163, 131, 39, 4, 35, 22, 261, 195, 1234, 222, 1235, 1236, 4, 12, 200, 1237, 478, 1238, 1239, 426, 26, 126, 23, 28, 81, 39, 4, 162, 370, 195, 260, 196, 63, 163, 131, 20, 12, 8, 13, 6, 7, 35, 22, 66, 28, 2062, 10, 6, 7, 28, 2062, 13, 6, 7, 66, 25, 50, 31, 5, 223, 10, 6, 7, 163, 1767, 4, 65, 9, 60, 2063, 672, 33, 8, 25, 382, 31, 5, 223, 13, 6, 7, 2064, 4, 83, 13, 6, 7, 15, 13, 6, 7, 47, 1183, 978, 4, 360, 2065, 62, 1319, 2066, 4, 46, 43, 10, 6, 7, 46, 4, 134, 23] ['全款', '时才', '右侧', '尾灯', '间', '一小', '点掉', '修补', '常见', '补漆笔', '恩', '很快', '℃']
            abstract_words ['你好', '，', '像', '这种', '情况', '确实', '不好', '说', '，', '你', '如果', '不', '放心', '的话', '，', '直接', '开去', '外面', '比较', '专业', '的', '修理厂', '，', '让', '专业', '的', '喷漆', '师傅', '帮', '你', '检查一下', '，', '如果', '确定', '四', 's', '店', '进行', '重新', '喷漆', '的话', '，', '这种', '情况', '属于', '四', 'ｓ', '存在', '消费', '欺诈', '，', '可以', '走', '法律', '程序', '退一', '赔', '三', '如果', '感觉', '问题', '不', '大', '的话', '，', '直接', '要求', '四', 's', '店', '进行', '重新', '喷漆', '就', '可以', '。']
            abs_ids [15, 4, 278, 35, 22, 235, 167, 6, 4, 18, 26, 28, 488, 39, 4, 162, 1233, 677, 51, 489, 5, 106, 4, 353, 489, 5, 131, 182, 254, 18, 36, 4, 26, 199, 195, 260, 196, 63, 163, 131, 39, 4, 35, 22, 261, 195, 1234, 222, 1235, 1236, 4, 12, 200, 1237, 478, 1238, 1239, 426, 26, 126, 23, 28, 81, 39, 4, 162, 370, 195, 260, 196, 63, 163, 131, 20, 12, 8]
            dec_inp: [2, 15, 4, 278, 35, 22, 235, 167, 6, 4, 18, 26, 28, 488, 39, 4, 162, 1233, 677, 51, 489, 5, 106, 4, 353, 489, 5, 131, 182, 254, 18, 36, 4, 26, 199, 195, 260, 196, 63, 163] 40
            target [15, 4, 278, 35, 22, 235, 167, 6, 4, 18, 26, 28, 488, 39, 4, 162, 1233, 677, 51, 489, 5, 106, 4, 353, 489, 5, 131, 182, 254, 18, 36, 4, 26, 199, 195, 260, 196, 63, 163, 131]
            sample_decoder_pad_mask [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            '''
            
            output = {
                "enc_len": enc_len,
                "enc_input": enc_input,
                "enc_input_extend_vocab": enc_input_extend_vocab,
                "article_oovs": article_oovs,
                "dec_input": dec_input,
                "target": target,
                "dec_len": dec_len,
                "article": article,
                "abstract": abstract,
                "abstract_sents": abstract_sentences,
                "sample_decoder_pad_mask": sample_decoder_pad_mask,
                "sample_encoder_pad_mask": sample_encoder_pad_mask,
            }
            yield output
    #测试数据处理
    if mode == "test":
        test_dataset = tf.data.TextLineDataset(test_x_path)
        #按条读取数据并数据规范化（转码、截断、转化为id、读取OOV并扩充词表）
        for raw_record in test_dataset:
            article = raw_record.numpy().decode("utf-8")
            article_words = article.split()[:max_enc_len]
            enc_len = len(article_words)

            enc_input = [vocab.word_to_id(w) for w in article_words]
            enc_input_extend_vocab, article_oovs = article_to_ids(article_words, vocab)

            sample_encoder_pad_mask = [1 for _ in range(enc_len)]

            output = {
                "enc_len": enc_len,
                "enc_input": enc_input,
                "enc_input_extend_vocab": enc_input_extend_vocab,
                "article_oovs": article_oovs,
                "dec_input": [],
                "target": [],
                "dec_len": 40,
                "article": article,
                "abstract": '',
                "abstract_sents": [],
                "sample_decoder_pad_mask": [],
                "sample_encoder_pad_mask": sample_encoder_pad_mask,
            }
            for _ in range(batch_size):
                yield output


#example_generator生成器函数根据不同的参数产生不同的数据，其实也就是根据指定是数据类型、维度等参数构建一个Dataset
#<MapDataset shapes: ({enc_input: (16, None), extended_enc_input: (16, None), article_oovs: (16, None), enc_len: (16,), article: (16,), max_oov_len: (), sample_encoder_pad_mask: (16, None)}, 
# {dec_input: (16, 40), dec_target: (16, 40), dec_len: (16,), abstract: (16,), sample_decoder_pad_mask: (16, 40)}), 
# types: ({enc_input: tf.int32, extended_enc_input: tf.int32, article_oovs: tf.string, enc_len: tf.int32, article: tf.string, max_oov_len: tf.int32, sample_encoder_pad_mask: tf.int32}, 
# {dec_input: tf.int32, dec_target: tf.int32, dec_len: tf.int32, abstract: tf.string, sample_decoder_pad_mask: tf.int32})>
def batch_generator(generator, vocab, train_x_path, train_y_path,
                    test_x_path, max_enc_len, max_dec_len, batch_size, mode):
    dataset = tf.data.Dataset.from_generator(lambda: generator(vocab, train_x_path, train_y_path, test_x_path,
                                                               max_enc_len, max_dec_len, mode, batch_size),
                                             output_types={
                                                 "enc_len": tf.int32,
                                                 "enc_input": tf.int32,
                                                 "enc_input_extend_vocab": tf.int32,
                                                 "article_oovs": tf.string,
                                                 "dec_input": tf.int32,
                                                 "target": tf.int32,
                                                 "dec_len": tf.int32,
                                                 "article": tf.string,
                                                 "abstract": tf.string,
                                                 "abstract_sents": tf.string,
                                                 "sample_decoder_pad_mask": tf.int32,
                                                 "sample_encoder_pad_mask": tf.int32,
                                             },
                                             output_shapes={
                                                 "enc_len": [],
                                                 "enc_input": [None],
                                                 "enc_input_extend_vocab": [None],
                                                 "article_oovs": [None],
                                                 "dec_input": [None],
                                                 "target": [None],
                                                 "dec_len": [],
                                                 "article": [],
                                                 "abstract": [],
                                                 "abstract_sents": [None],
                                                 "sample_decoder_pad_mask": [None],
                                                 "sample_encoder_pad_mask": [None],
                                             })
    #根据输入序列中的最大长度，自动的pad一个batch的序列

    dataset = dataset.padded_batch(batch_size,
                                   padded_shapes=({"enc_len": [],
                                                   "enc_input": [None],
                                                   "enc_input_extend_vocab": [None],
                                                   "article_oovs": [None],
                                                   "dec_input": [max_dec_len],
                                                   "target": [max_dec_len],
                                                   "dec_len": [],
                                                   "article": [],
                                                   "abstract": [],
                                                   "abstract_sents": [None],
                                                   "sample_decoder_pad_mask": [max_dec_len],
                                                   "sample_encoder_pad_mask": [None]}),
                                   padding_values={"enc_len": -1,
                                                   "enc_input": 1,
                                                   "enc_input_extend_vocab": 1,
                                                   "article_oovs": b'',
                                                   "dec_input": 1,
                                                   "target": 1,
                                                   "dec_len": -1,
                                                   "article": b'',
                                                   "abstract": b'',
                                                   "abstract_sents": b'',
                                                   "sample_decoder_pad_mask": 0,
                                                   "sample_encoder_pad_mask": 0},
                                   drop_remainder=True)

    def update(entry):
        return ({"enc_input": entry["enc_input"],
                 "extended_enc_input": entry["enc_input_extend_vocab"],
                 "article_oovs": entry["article_oovs"],
                 "enc_len": entry["enc_len"],
                 "article": entry["article"],
                 "max_oov_len": tf.shape(entry["article_oovs"])[1],
                 "sample_encoder_pad_mask": entry["sample_encoder_pad_mask"]},

                {"dec_input": entry["dec_input"],
                 "dec_target": entry["target"],
                 "dec_len": entry["dec_len"],
                 "abstract": entry["abstract"],
                 "sample_decoder_pad_mask": entry["sample_decoder_pad_mask"]})

    dataset = dataset.map(update)
    return dataset


#数据分批、规范化、id化
def batcher(vocab, hpm):
    dataset = batch_generator(example_generator, vocab, hpm["train_seg_x_dir"], hpm["train_seg_y_dir"],
                              hpm["test_seg_x_dir"], hpm["max_enc_len"],
                              hpm["max_dec_len"], hpm["batch_size"], hpm["mode"])

    return dataset


if __name__ == '__main__':
    pass
