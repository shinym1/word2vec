import jieba
import os
import collections
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
import math
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def produce_pun(path):
    '''生成标点符号列表'''
    with open(os.path.join(path, 'cn_punctuation.txt'), 'r', encoding='utf-8', errors='ignore') as f:
        punction = f.read()
    punction = punction.replace('\n', '')
    return punction


def produce_stop(path):
    '''生成标点符号列表'''
    with open(os.path.join(path, 'cn_stopwords.txt'), 'r', encoding='utf-8', errors='ignore') as f:
        stop = f.read()
    stop = stop.replace('\n', '')
    return stop


def read_txt(file):
    '''读取原始文本'''
    with open(file, 'r', encoding='gbk', errors='ignore') as f:
        r_txt = f.read()
    r_txt = r_txt.replace('本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com',
                          '')
    return r_txt


def produce_words(r_txt, punction):
    '''生成以词为单元的列表，并去除标点符号或者停用词'''
    txt_words = []
    txt_words_len = 0
    for words in jieba.cut(r_txt):
        if (words not in punction) and (not words.isspace()):
            txt_words.append(words)
            txt_words_len += 1
    return txt_words, txt_words_len


def produce_word(r_txt, punction):
    '''生成以字为单元的列表，并去除标点符号或者停用词'''
    txt_words = []
    txt_words_len = 0
    for words in jieba.cut(r_txt):
        if (words not in punction) and (not words.isspace()):
            for char in words:
                txt_words.append(char)
                txt_words_len += 1
    return txt_words, txt_words_len


def get_bigram_tf(self, word):
    # 得到二元词的词频表
    bigram_tf = {}
    for i in range(len(word) - 1):
        bigram_tf[(word[i], word[i + 1])] = bigram_tf.get(
            (word[i], word[i + 1]), 0) + 1
    return bigram_tf


class Vocab:
    """文本词表，语料库，用来生成唯一词元的次序和频率"""
    def __init__(self, tokens=None):
        counter = collections.Counter(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # 未知词元的索引为0,连接上已经生成的词组，两个列表直接用加号即可连接，不需要append
        self.idx_to_token = ['<unk>']
        # 生成词元字典，键值为位置序号，对应着频率从高到低
        self.token_to_idx = {}
        # 把新的单词列表转化成词表
        for token, freq in self._token_freqs:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    # 输出tokens的单词位置序号，定义了class Vocab的类似索引操作
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            # 如果存在该键，就返回键值，否则则认为是不存在或已删除的词元，返回0
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    # 给定位置序号列表，输出对应的单词
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):  # 输出频率从高到低排序的词元
        return self._token_freqs


def Zipf_plot(word, char, path):
    '''绘制频率-序列图'''
    freqs_word = [freq for token, freq in word.token_freqs]
    freqs_char = [freq for token, freq in char.token_freqs]
    backend_inline.set_matplotlib_formats('svg')
    plt.rcParams['font.size'] = 14
    plt.figure(figsize=(10, 6))
    plt.plot(freqs_word, label='word')
    plt.plot(freqs_char, label='character')
    plt.xlabel('index')
    plt.ylabel('frequency:n')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(path, 'Zipf.png'), dpi=300)
    plt.show()


def Zipf_word(path):
    '''齐普夫定律的验证结果'''
    txt_words = []
    punction = produce_pun(path)
    for file in os.listdir(os.path.join(path, 'txt')):
        r_txt = read_txt(os.path.join(path, 'txt', file))
        tem_txt_words, txt_words_len = produce_word(r_txt, punction=punction)
        txt_words.extend(tem_txt_words)
    return txt_words




def Zipf_words(path):
    '''齐普夫定律的验证结果'''
    txt_words = []
    punction = produce_pun(path)
    for file in os.listdir(os.path.join(path, 'txt')):
        r_txt = read_txt(os.path.join(path, 'txt', file))
        tem_txt_words, txt_words_len = produce_words(r_txt, punction=punction)
        txt_words.extend(tem_txt_words)
    return txt_words


def produce_Zipf_fig(path):
    word = Vocab(Zipf_words(path))
    char = Vocab(Zipf_word(path))
    Zipf_plot(word, char, path)


def entropy_one_words(file, path):
    '''一元词模型的信息熵'''
    r_txt = read_txt(os.path.join(path, 'txt', file))
    file = file.replace('.txt', '')
    stop = produce_stop(path)
    txt_words, txt_words_len = produce_words(r_txt, punction=stop)
    vocab = Vocab(txt_words)
    words_tf = vocab.token_freqs
    words_len = sum([freq for _, freq in words_tf])
    entropy = sum(
        [-(words[1] / words_len) * math.log(words[1] / words_len, 2) for words in
         words_tf])
    # print("<{}>基于词的一元模型的熵为：{}".format(file, entropy))
    return entropy


# def entropy_double_words(file, path):
#     '''二元词模型的信息熵'''
#
#     word_tf = get_bigram_tf(word)
#     last_word_tf = get_unigram_tf(word)
#     bigram_len = sum([item[1] for item in word_tf.items()])
#     entropy = []
#     for bigram in word_tf.items():
#         p_xy = bigram[1] / bigram_len  # 联合概率p(xy)
#         p_x_y = bigram[1] / last_word_tf[bigram[0][0]]  # 条件概率p(x|y)
#         entropy.append(-p_xy * math.log(p_x_y, 2))
#     entropy = sum(entropy)
#     if is_ci:
#         print("<{}>基于词的二元模型的中文信息熵为：{}比特/词".format(self.name, entropy))
#     else:
#         print("<{}>基于字的二元模型的中文信息熵为：{}比特/字".format(self.name, entropy))
#     return entropy


def entropy_one_word(file, path):
    '''一元词模型的信息熵'''
    r_txt = read_txt(os.path.join(path, 'txt', file))
    file = file.replace('.txt', '')
    stop = produce_stop(path)

    txt_word, txt_words_len = produce_word(r_txt, punction=stop)

    vocab = Vocab(txt_word)
    words_tf = vocab.token_freqs
    words_len = sum([freq for _, freq in words_tf])
    entropy = sum(
        [-(words[1] / words_len) * math.log(words[1] / words_len, 2) for words in
         words_tf])
    # print("<{}>基于字的一元模型的熵为：{}".format(file, entropy))
    return entropy

def bar_plot(x, y, path):
    # 创建柱状图
    plt.figure(figsize=(8, 6))
    plt.rcParams['font.sans-serif'] = ['SimSun']
    plt.rcParams['font.size'] = 12
    # plt.rcParams['font.family'] = 'Times New Roman'
    plt.xticks(rotation=45)
    bar_width = 0.25
    # 计算第二组柱状图的位置
    bar_positions1 = np.arange(16)-bar_width/2
    bar_positions2 = bar_positions1 + bar_width
    # 绘制第一组柱状图
    plt.bar(bar_positions1, y[0], width=bar_width, label='word', color='purple')
    for i, v in enumerate(y[0]):
        plt.text(i, v, str(round(v, 1)), ha='right', va='bottom', fontsize=10)
    # 绘制第二组柱状图
    plt.bar(bar_positions2, y[1], width=bar_width, label='character', color='orange')
    for i, v in enumerate(y[1]):
        plt.text(i, v, str(round(v, 1)), ha='left', va='bottom', fontsize=10)
    # 添加标题和标签
    plt.title("Histogram")
    plt.ylabel("entropy/bit")
    plt.ylim(0, 16)
    plt.xticks([i for i in range(16)], x)
    plt.tight_layout()
    plt.legend()
    # 显示图形
    plt.savefig(os.path.join(path, 'entropy.png'), dpi=300)
    plt.show()


def produce_hemi(path):
    fre_word = np.array([])
    fre_char = np.array([])
    for file in os.listdir(os.path.join(path, 'txt')):
        fre_word = np.append(fre_word, entropy_one_words(file, path))
    for file in os.listdir(os.path.join(path, 'txt')):
        fre_char = np.append(fre_char, entropy_one_word(file, path))
    fre = np.stack((fre_word, fre_char))
    x = os.listdir(os.path.join(path, 'txt'))
    for i in range(16):
        x[i] = x[i].replace('.txt', '')
    bar_plot(x, fre, path)




# path = './novel_set'

# produce_Zipf_fig(path)


# 数据量比较大，需要一定时间
#验证Zipf's law
# produce_hemi(path)  #calculate entropy


# path = './'
#
# # 数据和标签
# channels = ['通道 0', '通道 1', '通道 3', '通道 6', '通道 7']
# f_scores = [6.86, 6.81, 1.89, 3.48, 5.92]
#
# # 创建柱状图
# plt.figure(figsize=(8, 6))
# plt.rcParams['font.sans-serif'] = ['SimSun']
# plt.rcParams['font.size'] = 14
# plt.bar(channels, f_scores, width=0.4)
# for i, v in enumerate(f_scores):
#     plt.text(i, v, str(round(v, 1)), ha='center', va='bottom')

# 添加标题和标签
# plt.title('各通道信号的F值分数')
# plt.xlabel('信号通道')
# plt.ylabel('F值')
# plt.savefig('f.png', dpi=600)
# # 显示图表
# plt.show()