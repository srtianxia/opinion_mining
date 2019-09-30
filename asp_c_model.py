# @author     : srtianxia
# @time       : 2019/9/23 8:20
# @description:

import codecs
import json
import os
from collections import Counter

import keras.backend as K
import numpy as np
import pandas as pd
from keras import Input, Model
from keras.callbacks import Callback
from keras.layers import Lambda, Concatenate, RepeatVector, Dense, GlobalAveragePooling1D, Embedding
from keras.losses import categorical_crossentropy
from keras.utils import to_categorical
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras_bert.layers import MaskedConv1D
from keras_contrib.layers import CRF
from tqdm import tqdm
from keras_bert import AdamWarmup, calc_train_steps
import random
from pyhanlp import HanLP

random.seed(200)
tqdm.pandas()
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
np.random.seed(2019)

root_path = 'bert_base/pretraining_output/'  # 使用laptop的语料微调过的bert模型
config_path = root_path + 'bert_config.json'
checkpoint_path = root_path + 'model.ckpt-10000'
dict_path = root_path + 'vocab.txt'
MAX_LEN = 50
TAG_O = '<O>'

TEST_SIZE = 0.2
BATCH_SIZE = 64


def generate_cp_label(labels):
    cps_list = Counter(labels).most_common()
    cp2id = ({t: i for i, (t, c) in enumerate(cps_list)})
    id2cp = {v: k for k, v in cp2id.items()}
    return cp2id, id2cp


def encode_aspect_seq(maxlen=48):
    id_list = df_review['id'].to_list()
    val_ids = random.sample(id_list, int(len(id_list) * 0.1))
    train_reviews = df_review[~df_review['id'].isin(val_ids)]
    val_reviews = df_review[df_review['id'].isin(val_ids)]
    print(train_reviews.shape[0], val_reviews.shape[0])
    train_labels = df_label[~df_label['id'].isin(val_ids)]
    val_labels = df_label[df_label['id'].isin(val_ids)]

    term_to_id = {
        "A-B": 1,
        "A-I": 2,
        "A-E": 3,
        "A-S": 4,
    }

    def encode_label(df_label: pd.DataFrame, df_reviews: pd.DataFrame):
        texts = []
        c_labels = []
        a_seqs = []
        o_seqs = []
        ids = []
        lf_seqs = []
        rt_seqs = []
        for _, row in tqdm(df_label.iterrows(), 'encode label'):
            id = row['id']
            ids.append(id)
            review = df_reviews.loc[df_reviews['id'] == id]['Reviews'].values.tolist()[0]
            texts.append(review)
            C = row['Categories']
            c_labels.append(C)
            seq_a = np.zeros((maxlen,), dtype=np.int32)
            seq_o = np.zeros((maxlen,), dtype=np.int32)
            seq_lf = np.zeros((maxlen,), dtype=np.int32)
            seq_rt = np.zeros((maxlen,), dtype=np.int32)
            # for i_ in range(maxlen):
            #     seq_o[i_] = 1e-10
            e_o = row['O_end']
            opinion_terms = row['OpinionTerms']
            if opinion_terms != '_' and int(e_o) < maxlen:
                s_o = int(row['O_start'])
                e_o = int(e_o)
                for i in range(s_o, e_o):
                    seq_o[i] = 1
                for i in range(maxlen):
                    seq_lf[i] = abs(i - s_o)
                    seq_rt[i] = abs(i - e_o)
            lf_seqs.append(seq_lf)
            rt_seqs.append(seq_rt)
            o_seqs.append(seq_o)
            aspect_terms = row['AspectTerms']
            e = row['A_end']
            if aspect_terms != '_' and int(e) < maxlen:
                s = int(row['A_start'])
                e = int(e)
                if e - s == 1:  # 单个的
                    seq_a[s] = term_to_id["%s-S" % 'A']
                else:
                    seq_a[s] = term_to_id["%s-B" % 'A']
                    seq_a[e - 1] = term_to_id["%s-E" % 'A']
                    for p in range(s + 1, e - 1):
                        seq_a[p] = term_to_id["%s-I" % 'A']
            a_seqs.append(seq_a)
        return texts, a_seqs, o_seqs, lf_seqs, rt_seqs, c_labels, ids

    val_texts, val_a_seqs, val_o_seqs, val_lf_seqs, val_rt_seqs, val_cp_labels, val_ids = encode_label(val_labels,
                                                                                                       val_reviews)
    train_texts, train_a_seqs, train_o_seqs, train_lf_seqs, train_rt_seqs, train_cp_labels, train_ids = encode_label(
        train_labels, train_reviews)

    cp2id, id2cp = generate_cp_label(train_cp_labels + val_cp_labels)
    val_cp_labels = [cp2id[i] for i in val_cp_labels]
    train_cp_labels = [cp2id[i] for i in train_cp_labels]

    seq_id_val = np.array(val_ids)
    seq_id_train = np.array(train_ids)

    seq_A_val = np.stack(val_a_seqs)
    seq_A_train = np.stack(train_a_seqs)

    seq_A_all = np.concatenate((seq_A_val, seq_A_train))
    seq_A_all = to_categorical(seq_A_all)
    seq_A_val = seq_A_all[:len(seq_A_val)]
    seq_A_train = seq_A_all[len(seq_A_val):]

    seq_O_val = np.stack(val_o_seqs)
    seq_O_train = np.stack(train_o_seqs)

    seq_lf_val = np.stack(val_lf_seqs)
    seq_lf_train = np.stack(train_lf_seqs)

    seq_rt_val = np.stack(val_rt_seqs)
    seq_rt_train = np.stack(train_rt_seqs)

    seq_input_val, seq_seg_val = bert_text_to_seq(
        val_texts, tokenizer, maxlen=MAX_LEN
    )

    seq_input_train, seq_seg_train = bert_text_to_seq(
        train_texts, tokenizer, maxlen=MAX_LEN
    )

    id_to_term = dict([(v, k) for k, v in term_to_id.items()])

    all_cp_labels = train_cp_labels + val_cp_labels

    all_cp_labels = to_categorical(all_cp_labels)

    train_cp_labels = all_cp_labels[:len(train_cp_labels)]
    val_cp_labels = all_cp_labels[len(train_cp_labels):]

    return [seq_input_train, seq_seg_train, seq_A_train, seq_O_train, seq_lf_train, seq_rt_train, train_cp_labels,
            seq_id_train], \
           [seq_input_val, seq_seg_val, seq_A_val, seq_O_val, seq_lf_val, seq_rt_val, val_cp_labels, seq_id_val], (
               cp2id, id2cp), id_to_term, val_ids


def bert_text_to_seq(texts, tokenizer, maxlen=48):
    input_ids = []
    seg_ids = []
    for idx, text in tqdm(enumerate(texts), 'bert_text_to_seq'):
        ids, segs = tokenizer.encode(text, max_len=maxlen)
        input_ids.append(ids)
        seg_ids.append(segs)
    return np.array(input_ids), np.array(seg_ids)


class BertTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R


def pos_tag(review, maxlen=MAX_LEN):
    pos_results = HanLP.segment(review)

    tag_pos = 0
    postag = [TAG_O] * len(review)

    pos_tag_pos = []
    for idx, term in enumerate(pos_results):
        word = term.word
        tag = str(term.flag)
        words_len = len(word)
        pos_tag_pos.append((tag_pos, tag_pos + words_len, tag))
        tag_pos += words_len

    for (s, e, label) in pos_tag_pos:
        if e - s == 1:  # 单个的
            postag[s] = "%s-S" % label
        else:
            postag[s] = "%s-B" % label
            postag[e - 1] = "%s-E" % label
            for p in range(s + 1, e - 1):
                postag[p] = "%s-I" % label
    return postag + [TAG_O] * (maxlen - len(postag)) if len(postag) < maxlen else postag[:maxlen]


def decode_asp_c(seq_id, seq_O, text_review):
    max_len = seq_O.shape[1]
    seq_idx = np.arange(max_len)
    assert seq_O.shape[0] == len(text_review)
    opinions = []
    for id, s_ao, text in zip(seq_id, seq_O, text_review):
        idx_ob = seq_idx[np.where(s_ao == 1, True, False)]
        idx_oe = seq_idx[np.where(s_ao == 3, True, False)]
        idx_oi = seq_idx[np.where(s_ao == 4, True, False)]

        o_terms = []
        for i_b, i_e in zip(idx_ob, idx_oe):
            if i_b >= i_e + 1:
                continue
            o_terms.append(text[i_b: i_e + 1])

        for i_i in idx_oi:
            o_terms.append(text[i_i: i_i + 1])

        opinions.append((id, o_terms))
    return opinions


class Evaluation(Callback):
    def __init__(self, val_data, interval=1):
        self.val_data = val_data
        self.interval = interval
        self.best_f1 = 0.

    def on_epoch_end(self, epoch, log={}):
        if epoch % self.interval == 0:
            a_out, cp_out = self.model.predict(self.val_data[:-1], batch_size=BATCH_SIZE)
            # o_pred = np.argmax(o_out, axis=2)
            a_pred = np.argmax(a_out, axis=2)
            texts = [df_review[df_review['id'] == i]["Reviews"].values[0] for i in self.val_data[-1]]
            cp_pred = np.argmax(cp_out, -1)
            pred_vp_val = decode_asp_c(self.val_data[-1], a_pred, texts)
            cp_pred_decode = [id2cp[i] for i in cp_pred]
            true_df = df_label[df_label['id'].isin(val_ids)]
            pred_df = pd.DataFrame(pred_vp_val, columns=['id', 'AspectTerms'])
            pred_df['CP'] = cp_pred_decode
            S, P, G = 1e-10, 1e-10, 1e-10
            p_save_list = []
            t_save_list = []

            for (_, trues), (_, preds) in zip(true_df.groupby('id'), pred_df.groupby('id')):
                assert trues.shape[0] == preds.shape[0]
                id = trues['id'].values[0]
                R = set()
                T = set()
                for (_, true_row), (_, pred_row) in zip(trues.iterrows(), preds.iterrows()):
                    T.add(
                        (true_row['OpinionTerms'], true_row['AspectTerms'], true_row['Categories'],
                         )
                    )
                    CP = pred_row['CP']
                    aspect_terms = pred_row['AspectTerms']
                    if len(aspect_terms) == 0:
                        aspect_terms = '_'
                    else:
                        aspect_terms = aspect_terms[0]
                    R.add(
                        (true_row['OpinionTerms'], aspect_terms, CP)
                    )
                S += len(R & T)
                P += len(R)
                G += len(T)
                for i in R:
                    p_save_list.append({
                        'id': str(id),
                        'OpinionTerms': i[0],
                        'AspectTerms': i[1],
                        'Categories': i[2],

                    })
                for i in T:
                    t_save_list.append({
                        'id': str(id),
                        'OpinionTerms': i[0],
                        'AspectTerms': i[1],
                        'Categories': i[2],

                    })
            with codecs.open(f'./data/ea/dev_pred_{epoch}.json', 'w', encoding='utf-8') as f:  # 错误分析
                json.dump(p_save_list, f, indent=4, ensure_ascii=False)

            with codecs.open(f'./data/ea/dev_true_{epoch}.json', 'w', encoding='utf-8') as f:
                json.dump(t_save_list, f, indent=4, ensure_ascii=False)

            precision, recall = S / P, S / G
            f1 = 2 * precision * recall / (precision + recall)
            if f1 > self.best_f1:
                self.best_f1 = f1
                train_model.save_weights('finetune_weight/best_f1/best_f1_mix_c.h5')

            print(
                f'precision = {precision}',
                f'recall = {recall}',
                f'f1 = {f1}',
                "\n",
            )


def generate_postag2id(train_pos_tag):
    reviews_list = []
    for sent in train_pos_tag:
        reviews_list.extend(sent)
    ct_list = Counter(reviews_list).most_common()
    postag2id = ({t: i for i, (t, c) in enumerate(ct_list)})
    return postag2id


def dilated_gated_conv1d(seq, mask, name, dilation_rate=1):
    """膨胀门卷积（残差式）
    """
    dim = K.int_shape(seq)[-1]
    h = MaskedConv1D(filters=dim * 2, kernel_size=3, padding='same', dilation_rate=dilation_rate, name=name)(seq)

    def _gate(x):
        dropout_rate = 0.1
        s, h = x
        g, h = h[:, :, :dim], h[:, :, dim:]
        g = K.in_train_phase(K.dropout(g, dropout_rate, seed=2019), g)
        g = K.sigmoid(g)
        return g * s + (1 - g) * h

    seq = Lambda(_gate)([seq, h])
    seq = Lambda(lambda x: x[0] * x[1])([seq, mask])
    return seq


df_review = pd.read_csv('./data_phase2/Train_makeup_reviews.csv')
df_label = pd.read_csv('./data_phase2/Train_makeup_labels.csv')

token_dict = {}
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

tokenizer = BertTokenizer(token_dict)
train_data, val_data, (cp2id, id2cp), id_to_term, val_ids = encode_aspect_seq(maxlen=MAX_LEN)

bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)

for l in bert_model.layers:
    l.trainable = True

x1_in = Input(shape=(MAX_LEN,), name='x1_in')
x2_in = Input(shape=(MAX_LEN,), name='x2_in')
opinion_mask_in = Input(shape=(MAX_LEN,), name='opinion_mask_in')
lf_pos_in = Input(shape=(MAX_LEN,), name='lf_pos_in')
rt_pos_in = Input(shape=(MAX_LEN,), name='rt_pos_in')
seq_a_in = Input(shape=(MAX_LEN, len(id_to_term) + 1), name='seq_a_in')
c_in = Input(shape=(len(cp2id),), name='c_in')

opinion_mask = Lambda(lambda x: K.expand_dims(K.cast(K.greater(x, 0), 'float32'), axis=1))(opinion_mask_in)
opinion_mask_emb = Lambda(lambda x: K.cast(K.greater(x, 0), 'float32'))(opinion_mask_in)
pos_tag = Embedding(2, 10, name='embpos')(opinion_mask_emb)
lf_pos_tag = Embedding(MAX_LEN, 10, name='lf_embpos')(lf_pos_in)
rt_pos_tag = Embedding(MAX_LEN, 10, name='rt_embpos')(rt_pos_in)

x = bert_model([x1_in, x2_in])

opinion_vec = Lambda(lambda x: K.batch_dot(x[0], x[1]) / K.sum(x[0], keepdims=True))([opinion_mask, x])  # [?,1,768]
opinion_vec_ori = Lambda(lambda x: K.squeeze(x, axis=1))(opinion_vec)
opinion_vec = RepeatVector(MAX_LEN)(opinion_vec_ori)
mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x1_in)

x = Concatenate()([x, opinion_vec, pos_tag, lf_pos_tag, rt_pos_tag])

x = dilated_gated_conv1d(x, mask, 'CNN_1', 1)
x = dilated_gated_conv1d(x, mask, 'CNN_2', 2)
x = dilated_gated_conv1d(x, mask, 'CNN_3', 5)

crf = CRF(len(id_to_term) + 1)
a_out = crf(x)
loss_A = crf.loss_function(seq_a_in, a_out)  # 直接加入 Lambda层后 计算图会出错
loss_A = Lambda(lambda x: K.mean(x))(loss_A)

x = GlobalAveragePooling1D()(x)
x = Concatenate()([x, opinion_vec_ori])

c_out = Dense(len(cp2id), activation='softmax', name='cp_out_Dense')(x)

a_model = Model([x1_in, x2_in, opinion_mask_in, lf_pos_in, rt_pos_in], a_out)
cp_model = Model([x1_in, x2_in, opinion_mask_in, lf_pos_in, rt_pos_in], c_out)

train_model = Model([x1_in, x2_in, seq_a_in, opinion_mask_in, lf_pos_in, rt_pos_in, c_in], [a_out, c_out])

loss_c = Lambda(lambda x: K.mean(categorical_crossentropy(x[0], x[1])), name='loss_p')([c_in, c_out])

train_model.add_loss(loss_A)
train_model.add_loss(loss_c)

total_steps, warmup_steps = calc_train_steps(
    num_example=train_data[0].shape[0],
    batch_size=BATCH_SIZE,
    epochs=100,
    warmup_proportion=0.05,
)

optimizer = AdamWarmup(total_steps, warmup_steps, lr=1e-4, min_lr=1e-6)

train_model.compile(optimizer=optimizer)

train_model.metrics_tensors.append(loss_A)
train_model.metrics_names.append('loss_A')
train_model.metrics_tensors.append(loss_c)
train_model.metrics_names.append('loss_c')
train_model.summary()

eval_callback = Evaluation(val_data=val_data)

train_model.fit(train_data[:-1], epochs=100, shuffle=True, batch_size=BATCH_SIZE, callbacks=[eval_callback])
