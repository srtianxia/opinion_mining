# @author     : srtianxia
# @time       : 2019/9/22 20:45
# @description:

import codecs
import os
import pickle

import keras.backend as K
import numpy as np
import pandas as pd
from keras import Input, Model
from keras.callbacks import Callback
from keras.layers import Embedding, Concatenate, Dense, Lambda
from keras.losses import categorical_crossentropy
from keras.utils import to_categorical
from keras_bert import Tokenizer, load_trained_model_from_checkpoint
from keras_contrib.layers import CRF
from pyhanlp import HanLP
from sklearn.model_selection import train_test_split
from tensorflow import set_random_seed
from tqdm import tqdm
from keras_bert import AdamWarmup, calc_train_steps

tqdm.pandas()
set_random_seed(2019)

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

np.random.seed(2019)
MAX_LEN = 48
BATCH_SIZE = 64
TEST_SIZE = 0.1
LEARNING_RATE = 1e-5
EPOCHS = 100
POS_TAG_DIM = 256

TAG_PAD = '<PAD>'

df_review_laptop = pd.read_csv('./data_phase2/Train_laptop_reviews.csv')
df_label_laptop = pd.read_csv('./data_phase2/Train_laptop_labels.csv')

df_review_makeup = pd.read_csv('./data_phase2/Train_makeup_reviews.csv')
df_label_makeup = pd.read_csv('./data_phase2/Train_makeup_labels.csv')

nums_laptop = df_review_laptop.shape[0]
df_review_makeup['id'] = df_review_makeup['id'] + nums_laptop
df_label_makeup['id'] = df_label_makeup['id'] + nums_laptop

df_review = pd.concat([df_review_laptop, df_review_makeup], ignore_index=True)
df_label = pd.concat([df_label_laptop, df_label_makeup], ignore_index=True)


def decode_seq(seq_id, seq_O, seq_P, id_to_label, text_review):
    max_len = seq_O.shape[1]
    seq_idx = np.arange(max_len)
    assert seq_O.shape[0] == seq_P.shape[0] == len(text_review)
    viewpoints = []
    for id, s_ao, s_cp, text in tqdm(zip(seq_id, seq_O, seq_P, text_review), 'decode_seq'):
        idx_ob = seq_idx[np.where(s_ao == 1, True, False)]
        idx_oe = seq_idx[np.where(s_ao == 3, True, False)]
        idx_oi = seq_idx[np.where(s_ao == 4, True, False)]

        o_terms = []

        for i_b, i_e in zip(idx_ob, idx_oe):
            if i_b >= i_e + 1:
                continue
            label = max(s_cp[i_b: i_e + 1])
            o_terms.append((text[i_b: i_e + 1], id_to_label.get(label, 'O'), i_b, i_e + 1))

        for i_i in idx_oi:
            label = max(s_cp[i_i: i_i + 1])
            o_terms.append((text[i_i: i_i + 1], id_to_label.get(label, 'O'), i_i, i_i + 1))

        viewpoints.append((id, o_terms))
    return viewpoints


def encode_seq(df_label, maxlen=48):
    label_to_id = {
        '中性': 1,
        '负面': 2,
        '正面': 3  # 这样解码的时候可能更合理
    }

    term_to_id = {
        "O-B": 1,
        "O-I": 2,
        "O-E": 3,
        "O-S": 4,
    }

    def encode_term(pos, label):
        seq = np.zeros((maxlen,), dtype=np.int32)
        for (s, e) in pos:
            if e - s == 1:  # 单个的
                seq[s] = term_to_id["%s-S" % label]
            else:
                seq[s] = term_to_id["%s-B" % label]
                seq[e - 1] = term_to_id["%s-E" % label]
                for p in range(s + 1, e - 1):
                    seq[p] = term_to_id["%s-I" % label]
        return seq.reshape((1, -1))

    def encode_label(pos_o, label):
        seq = np.zeros((maxlen,), dtype=np.int32)
        for (s, e), l in zip(pos_o, label):
            if s == " " or int(e) >= maxlen:
                continue
            s = int(s)
            e = int(e)
            if e - s == 1:
                seq[s] = label_to_id[l]
            else:
                seq[s] = label_to_id[l]
                seq[e - 1] = label_to_id[l]
                for p in range(s + 1, e - 1):
                    seq[p] = label_to_id[l]
        return seq.reshape((1, -1))

    seq_O = df_label.groupby("id").apply(  # 所有的O
        lambda x: encode_term(
            [
                (int(s), int(e))
                for s, e in zip(x["O_start"], x["O_end"])
                if s != " " and int(e) < maxlen
            ],
            "O",
        )
    )

    seq_CP = df_label.groupby("id").apply(
        lambda x: encode_label(
            [(s, e) for s, e in zip(x["O_start"], x["O_end"])],
            [p for p in x["Polarities"]],
        )
    )

    seq_id = np.array(df_label.groupby("id").apply(lambda x: list(x['id'])[0]).to_list())
    seq_O = np.vstack(seq_O)

    seq_P = np.vstack(seq_CP)

    id_to_label = dict([(v, k) for k, v in label_to_id.items()])
    id_to_term = dict([(v, k) for k, v in term_to_id.items()])
    return seq_id, seq_O, seq_P, id_to_label, id_to_term


def cal_opinion_metrics(pred_vp, true_vp):
    true_df = pd.DataFrame(true_vp, columns=['id', 'o', 'p', 's', 'e'])
    S, P, G = 1e-10, 1e-10, 1e-10
    pred_df = pd.DataFrame(pred_vp, columns=['id', 'o_pred'])
    for idx, trues in tqdm(true_df.groupby('id'), 'cal_opinion_metrics'):
        id = trues['id'].values[0]
        T = set()
        for _, true_row in trues.iterrows():
            T.add((true_row['o'], true_row['p'], str(true_row['s']), str(true_row['e'])))
        pred_list = pred_df.loc[pred_df['id'] == id]['o_pred'].values.tolist()[0]
        R = set()
        for pred_row in pred_list:
            R.add((pred_row[0], pred_row[1], str(pred_row[2]), str(pred_row[3])))
        S += len(R & T)
        P += len(R)
        G += len(T)

    precision, recall = S / P, S / G
    f1 = 2 * precision * recall / (precision + recall)

    print(
        f'precision = {precision}',
        f'recall = {recall}',
        f'f1 = {f1}',
        "\n",
    )
    return precision, recall, f1


def split_viewpoints(seq_id, seq_input, seq_mask, seq_AO, seq_P, seq_postag):
    idx = np.random.permutation(range(seq_id.shape[0]))
    tr_idx, te_idx = train_test_split(idx, test_size=TEST_SIZE, random_state=2019)
    return (
        [
            seq_id[tr_idx],
            seq_input[tr_idx],
            seq_mask[tr_idx],
            seq_postag[tr_idx],
            seq_AO[tr_idx],
            seq_P[tr_idx]
        ],
        [
            seq_id[te_idx],
            seq_input[te_idx],
            seq_mask[te_idx],
            seq_postag[te_idx],
            seq_AO[te_idx],
            seq_P[te_idx]
        ],
    )


root_path = './chinese_L-12_H-768_A-12/'
config_path = root_path + 'bert_config.json'
checkpoint_path = root_path + 'bert_model.ckpt'
dict_path = root_path + 'vocab.txt'

token_dict = {}
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


class BertTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')
            else:
                R.append('[UNK]')
        return R


def bert_text_to_seq(texts, tokenizer, maxlen=48):
    input_ids = []
    seg_ids = []
    for idx, text in tqdm(enumerate(texts), 'bert_text_to_seq'):
        ids, segs = tokenizer.encode(text, max_len=maxlen)
        input_ids.append(ids)
        seg_ids.append(segs)
    return np.array(input_ids), np.array(seg_ids)


def pos_tag(review, maxlen=MAX_LEN):
    pos_results = HanLP.segment(review)
    tag_pos = 0
    postag = [TAG_PAD] * len(review)
    pos_tag_pos = []
    for idx, term in enumerate(pos_results):
        word = term.word
        tag = str(term.nature)
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
    return postag + [TAG_PAD] * (maxlen - len(postag)) if len(postag) < maxlen else postag[:maxlen]


def main():
    seq_id, seq_O, seq_P, id_to_label, id_to_term = encode_seq(df_label=df_label, maxlen=MAX_LEN)

    class Evaluation(Callback):
        def __init__(self, val_data, interval=1):
            self.val_data = val_data
            self.interval = interval
            self.best_f1 = 0.

            self.true_vp_val = [
                (
                    row["id"],
                    row["OpinionTerms"],
                    row["Polarities"],
                    row['O_start'],
                    row['O_end']
                )
                for rowid, row in df_label[df_label['id'].isin(self.val_data[0])].iterrows()
            ]

        def on_epoch_end(self, epoch, log={}):
            if epoch % self.interval == 0:
                o_out, p_out = pred_model.predict(self.val_data[1:4], batch_size=BATCH_SIZE)  # CRF概率
                o_pred = np.argmax(o_out, axis=2)
                p_pred = np.argmax(p_out, axis=2)

                texts = [df_review[df_review['id'] == i]["Reviews"].values[0] for i in self.val_data[0]]

                pred_vp_val = decode_seq(
                    self.val_data[0], o_pred, p_pred, id_to_label, texts)

                precision, recall, f1 = cal_opinion_metrics(pred_vp_val, self.true_vp_val)
                if f1 > self.best_f1:
                    self.best_f1 = f1
                    self.model.save_weights(f'./model_op/op_model_0924_viteb.weights')
                    print(f'best = {f1}')

    tokenizer = BertTokenizer(token_dict)

    seq_input, seq_seg = bert_text_to_seq(
        list(df_review["Reviews"]), tokenizer, maxlen=MAX_LEN
    )

    true_vp = [
        (
            row["id"],
            row["OpinionTerms"],
            row["Polarities"],
            row['O_start'],
            row['O_end']
        )
        for rowid, row in df_label.iterrows()
    ]

    pred_vp = decode_seq(
        seq_id, seq_O, seq_P, id_to_label, list(df_review["Reviews"])
    )

    cal_opinion_metrics(pred_vp, true_vp)

    seq_O = to_categorical(seq_O)

    seq_P = to_categorical(seq_P)

    df_review['pos_tag'] = df_review['Reviews'].progress_apply(pos_tag)

    with open('./data/postag2id_0922_laptop_make_up.pkl', 'rb') as f:
        postag2id = pickle.load(f)

    df_review['pos_tag'] = df_review['pos_tag'].progress_apply(lambda postag: [postag2id[x] for x in postag])

    seq_postag = np.array(df_review['pos_tag'].values.tolist())

    view_train, view_val = split_viewpoints(seq_id, seq_input, seq_seg, seq_O, seq_P, seq_postag)

    print(view_val[0])
    print('------------------- 保存验证集的id ---------------------')
    print('保存final 验证集的val ids')

    # np.save('./data/final_makeup_laptop_val_ids', view_val[0])
    print('------------------- 保存完毕 ---------------------------')
    # exit()
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)
    for l in bert_model.layers:
        l.trainable = True

    x1_in = Input(shape=(MAX_LEN,), name='x1_in')
    x2_in = Input(shape=(MAX_LEN,), name='x2_in')
    o_in = Input(shape=(MAX_LEN, len(id_to_term) + 1,), name='o_in')
    p_in = Input(shape=(MAX_LEN, len(id_to_label) + 1,), name='p_in')

    pos_tag_in = Input(shape=(MAX_LEN,), name='pos_tag_in')
    pos_tag_emb = Embedding(len(postag2id), POS_TAG_DIM, trainable=True)(pos_tag_in)

    x = bert_model([x1_in, x2_in])
    x = Concatenate()([x, pos_tag_emb])

    p_out = Dense(len(id_to_label) + 1, activation='softmax')(x)  # p_out 是极性的输出
    crf = CRF(len(id_to_term) + 1)
    o_out = crf(x)
    loss_seq_O = crf.loss_function(o_in, o_out)  # 直接加入 Lambda层后 计算图会出错
    loss_seq_O = Lambda(lambda x: K.mean(x))(loss_seq_O)
    # loss_seq_O = Lambda(lambda x: K.mean(categorical_crossentropy(x[0], x[1])), name='loss_seq_O')([o_in, o_out])

    loss_p = Lambda(lambda x: K.mean(categorical_crossentropy(x[0], x[1])), name='loss_c')([p_in, p_out])

    train_model = Model([x1_in, x2_in, pos_tag_in, o_in, p_in], [o_out, p_out])
    pred_model = Model([x1_in, x2_in, pos_tag_in], [o_out, p_out])
    train_model._losses = []
    train_model._per_input_losses = {}
    train_model.add_loss(loss_seq_O)
    train_model.add_loss(loss_p)

    print(view_train[0].shape[0])

    total_steps, warmup_steps = calc_train_steps(
        num_example=view_train[0].shape[0],
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        warmup_proportion=0.1,
    )
    # optimizer = Adam(lr=1e-5)
    optimizer = AdamWarmup(total_steps, warmup_steps, lr=5e-5, min_lr=1e-6)

    train_model.compile(optimizer=optimizer)
    train_model.metrics_tensors.append(loss_seq_O)
    train_model.metrics_names.append('loss_seq_O')
    train_model.metrics_tensors.append(loss_p)
    train_model.metrics_names.append('loss_p')
    train_model.summary()

    eval_callback = Evaluation(val_data=view_val)

    train_model.fit(view_train[1:], epochs=EPOCHS, shuffle=True, batch_size=BATCH_SIZE, callbacks=[eval_callback])


if __name__ == "__main__":
    main()
