from collections import defaultdict
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def save_word_dict(vocab, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for line in vocab:
            w, i = line
            f.write("%s\t%d\n" % (w, i))


def read_data(path_1, path_2, path_3):
    with open(path_1, 'r', encoding='utf-8') as f1, \
            open(path_2, 'r', encoding='utf-8') as f2, \
            open(path_3, 'r', encoding='utf-8') as f3:
        words = []
        # print(f1)
        for line in f1:
            words += line.split()

        for line in f2:
            words += line.split(' ')

        for line in f3:
            words += line.split(' ')

    return words


def build_vocab(items, sort=True, min_count=0, lower=False):
    """
    构建词典列表
    :param items: list  [item1, item2, ... ]
    :param sort: 是否按频率排序，否则按items排序
    :param min_count: 词典最小频次
    :param lower: 是否小写
    :return: list: word set
    """
    result = []
    if sort:
        # sort by count
        dic = defaultdict(int)
        for item in items:
            for i in item.split(" "):
                i = i.strip()#去除每个词带着的空格
                if not i: continue
                i = i if not lower else item.lower()
                dic[i] += 1
        # print(dic)
        # sort
        #按照字典里的词频进行排序，出现次数多的排在前面
        dic_order=sorted(dic.items(),key=lambda x:x[1],reverse=True) 
        #dic_order=[('，', 23), ('说', 18), ('：', 16),]
        for i, item in enumerate(dic_order):
            key = item[0]
            #频次大于等于预设值的才保留
            if min_count and min_count > item[1]:
                continue
            result.append(key)
    else:
        # sort by items
        '''
        for item in items:
            for i in item.split(" "):
                i = i.strip()#去除每个词带着的空格
                if not i: continue
                i = i if not lower else item.lower()
                result.append(item)
        '''
        
        for i, item in enumerate(items):
            i = i.strip()#去除每个词带着的空格
            item = item if not lower else item.lower()
            result.append(item)
        
        #词表去重
        result=list(set(result))
    
    #建立项目的vocab和reverse_vocab，vocab的结构是（词，index）
    vocab= [(word,index)for index,word in enumerate(result)]
    reverse_vocab = [(index,word)for index,word in enumerate(result)]
    return vocab, reverse_vocab


if __name__ == '__main__':
    lines = read_data('{}/datasets/train_set.seg_x.txt'.format(BASE_DIR),
                      '{}/datasets/train_set.seg_y.txt'.format(BASE_DIR),
                      '{}/datasets/test_set.seg_x.txt'.format(BASE_DIR))
    vocab, reverse_vocab = build_vocab(lines)
    save_word_dict(vocab, '{}/datasets/vocab.txt'.format(BASE_DIR))