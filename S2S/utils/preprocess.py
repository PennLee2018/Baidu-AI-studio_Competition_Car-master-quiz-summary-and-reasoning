import numpy as np
import pandas as pd
import re
from jieba import posseg
import jieba
from tokenizer import segment
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


REMOVE_WORDS = ['|', '[', ']', '语音', '图片', ' ']


def read_stopwords(path):
    lines = set()
    with open(path, mode='r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            lines.add(line)
    return lines


def remove_words(words_list):
    words_list = [word for word in words_list if word not in REMOVE_WORDS]
    return words_list


def parse_data(train_path, test_path):
    #读取原始数据集
    train_df = pd.read_csv(train_path, encoding='utf-8')
    #查看数据信息及表头,一共6列
    #print(train_df.info())
    #print(train_df.head())
    #去掉'Report'这一列的na值，any是带缺失值的所有行，'all’指清除全是缺失值的,True表示直接在原数据上更改
    train_df.dropna(subset=['Report'], how='any', inplace=True)
    
    #将其他列的缺失值填空格
    train_df.fillna('', inplace=True)
    
    #将Question和Dialogue拼接起来
    train_x = train_df.Question.str.cat(train_df.Dialogue)
    #print(train_x.head())
    print('train_x is ', len(train_x))
    train_x = train_x.apply(preprocess_sentence)
    #print(train_x[0])="方向机 重 ， 助力 泵 ， 方向机 都 换 了 还是 一...释 。 技师 说 ： 技师 说 ："
    print('train_x is ', len(train_x))
    train_y = train_df.Report
    print('train_y is ', len(train_y))
    train_y = train_y.apply(preprocess_sentence)
    #print('train_y is ', len(train_y))=82873
    #print(train_y[0])="随时 联系"
    # if 'Report' in train_df.columns:
        # train_y = train_df.Report
        # print('train_y is ', len(train_y))
        
    #测试集做同样的处理
    test_df = pd.read_csv(test_path, encoding='utf-8')
    test_df.fillna('', inplace=True)
    test_x = test_df.Question.str.cat(test_df.Dialogue)
    test_x = test_x.apply(preprocess_sentence)
    print('test_x is ', len(test_x))
    test_y = []
    train_x.to_csv('{}/datasets/train_set.seg_x.txt'.format(BASE_DIR), index=None, header=False)
    train_y.to_csv('{}/datasets/train_set.seg_y.txt'.format(BASE_DIR), index=None, header=False)
    test_x.to_csv('{}/datasets/test_set.seg_x.txt'.format(BASE_DIR), index=None, header=False)


def save_data(data_1, data_2, data_3, data_path_1, data_path_2, data_path_3, stop_words_path=''):
    stopwords = read_stopwords(stop_words_path)
    with open(data_path_1, 'w', encoding='utf-8') as f1:
        count_1 = 0
        for line in data_1:
            # print(line)
            if isinstance(line, str):
                seg_list = segment(line.strip(), cut_type='word')
                seg_list = remove_words(seg_list)
                # seg_words = []
                # for j in seg_list:
                #     if j in stopwords:
                #         continue
                #     seg_words.append(j)
                if len(seg_list) > 0:
                    seg_line = ' '.join(seg_list)
                    f1.write('%s' % seg_line)
                    f1.write('\n')
                    count_1 += 1
        print('train_x_length is ', count_1)

    with open(data_path_2, 'w', encoding='utf-8') as f2:
        count_2 = 0
        for line in data_2:
            if isinstance(line, str):
                seg_list = segment(line.strip(), cut_type='word')
                seg_list = remove_words(seg_list)
                # seg_words = []
                # for j in seg_list:
                #     if j in stopwords:
                #         continue
                #     seg_words.append(j)
                # if len(seg_list) > 0:
                seg_line = ' '.join(seg_list)
                f2.write('%s' % seg_line)
                f2.write('\n')
                count_2 += 1
        print('train_y_length is ', count_2)

    with open(data_path_3, 'w', encoding='utf-8') as f3:
        count_3 = 0
        for line in data_3:
            if isinstance(line, str):
                seg_list = segment(line.strip(), cut_type='word')
                seg_list = remove_words(seg_list)
                if len(seg_list) > 0:
                    seg_line = ' '.join(seg_list)
                    f3.write('%s' % seg_line)
                    f3.write('\n')
                    count_3 += 1
        print('test_y_length is ', count_3)


def preprocess_sentence(sentence):
    #分词
    seg_list = segment(sentence.strip(), cut_type='word')
    #去停用词
    seg_list = remove_words(seg_list)
    #拼接成句子,中间以空格分开    
    seg_line = ' '.join(seg_list)
    return seg_line


if __name__ == '__main__':
    # 需要更换成自己数据的存储地址
    parse_data('{}/datasets/AutoMaster_TrainSet.csv'.format(BASE_DIR),
               '{}/datasets/AutoMaster_TestSet.csv'.format(BASE_DIR))


