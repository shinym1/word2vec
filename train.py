import math
import torch
from torch import nn
from d2l import torch as d2l
from torch.utils import data
from de_Animator import Animator
import os
import law_veri
import jieba
from matplotlib import pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def produce_word(r_txt, punction):
    '''生成以词为单元的列表，并去除标点符号或者停用词'''
    txt_words = []
    for word in r_txt:
        if (word not in punction) and (not word.isspace()):
            txt_words.append(word)
    return txt_words


def load_data_ptb(batch_size, max_window_size, num_noise_words):
    """Download the PTB dataset and then load it into memory.

    Defined in :numref:`subsec_word2vec-minibatch-loading`"""
    # num_workers = d2l.get_dataloader_workers()
    # sentences = d2l.read_ptb()
    path = './novel_set'
    lines = []
    for file in os.listdir(os.path.join(path, 'txt')):
        tem_txt = law_veri.read_txt(os.path.join(path, 'txt', file))
        tem_txt = [line for line in tem_txt.split('。')]
        lines.extend(tem_txt)
    sentences = []
    punction = law_veri.produce_stop(path)
    lines = [jieba.cut(line) for line in lines]
    for line in lines:
        sentences.append(produce_word(line, punction))
    vocab = d2l.Vocab(sentences, min_freq=5)
    subsampled, counter = d2l.subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = d2l.get_centers_and_contexts(
        corpus, max_window_size)
    all_negatives = d2l.get_negatives(
        all_contexts, vocab, counter, num_noise_words)

    class PTBDataset(data.Dataset):
        def __init__(self, centers, contexts, negatives):
            assert len(centers) == len(contexts) == len(negatives)
            self.centers = centers
            self.contexts = contexts
            self.negatives = negatives

        def __getitem__(self, index):
            return (self.centers[index], self.contexts[index],
                    self.negatives[index])

        def __len__(self):
            return len(self.centers)

    dataset = PTBDataset(all_centers, all_contexts, all_negatives)
    data_iter = data.DataLoader(dataset, batch_size, shuffle=True,
                                      collate_fn=d2l.batchify)
                                      # num_workers=num_workers)
    return data_iter, vocab


# d2l.DATA_HUB['ptb'] = (d2l.DATA_URL + 'ptb.zip',
#     '319d85e578af0cdc590547f26231e4e31cdf1e42')
batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab = load_data_ptb(batch_size, max_window_size,
                                        num_noise_words)
embed = nn.Embedding(num_embeddings=20, embedding_dim=4)
# print(f'Parameter embedding_weight ({embed.weight.shape}, '
#     f'dtype={embed.weight.dtype})')


def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred


# print(skip_gram(torch.ones((2, 1), dtype=torch.long),
#     torch.ones((2, 4), dtype=torch.long), embed, embed).shape)


class SigmoidBCELoss(nn.Module):
    # 带掩码的二元交叉熵损失
    def __init__(self):
        super().__init__()

    def forward(self, inputs, target, mask=None):
        out = nn.functional.binary_cross_entropy_with_logits(
            inputs, target, weight=mask, reduction="none")
        return out.mean(dim=1)


loss = SigmoidBCELoss()


def sigmd(x):
    return -math.log(1 / (1 + math.exp(-x)))


embed_size = 100
net = nn.Sequential(nn.Embedding(num_embeddings=len(vocab),
embedding_dim=embed_size),
nn.Embedding(num_embeddings=len(vocab),
embedding_dim=embed_size))


def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
    def init_weights(m):
        if type(m) == nn.Embedding:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    animator = Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs])
    # 规范化的损失之和，规范化的损失数
    metric = d2l.Accumulator(2)
    for epoch in range(num_epochs):
        timer, num_batches = d2l.Timer(), len(data_iter)
        for i, batch in enumerate(data_iter):
            optimizer.zero_grad()
            center, context_negative, mask, label = [
                data.to(device) for data in batch]
            pred = skip_gram(center, context_negative, net[0], net[1])
            l = (loss(pred.reshape(label.shape).float(), label.float(), mask)
                 / mask.sum(axis=1) * mask.shape[1])
            l.sum().backward()
            optimizer.step()
            metric.add(l.sum(), l.numel())
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1],))
    plt.savefig('word2vec.png', dpi=600)
    print(f'loss {metric[0] / metric[1]:.3f}, '
          f'{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')


lr, num_epochs = 0.002, 5
# train(net, data_iter, lr, num_epochs)
# torch.save(net.state_dict(), 'word2vec.pth')


def get_similar_tokens(query_token, k, embed): #给定文本，找出前k个最大余弦相似度的词
    W = embed.weight.data
    x = W[vocab[query_token]]
    # 计算余弦相似性。增加1e-9以获得数值稳定性
    cos = torch.mv(W, x) / torch.sqrt(torch.sum(W * W, dim=1) *
        torch.sum(x * x) + 1e-9)
    topk = torch.topk(cos, k=k+1)[1].cpu().numpy().astype('int32')
    for i in topk[0:]: # 删除输入词
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}\t')

net.load_state_dict(torch.load('word2vec.pth'))
net.eval()
get_similar_tokens('汉子', 3, net[0])