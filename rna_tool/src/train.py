import os
from collections import Counter
import logging
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import scale
from sklearn.model_selection import StratifiedKFold
from tensorflow.python.ops.gen_array_ops import Reshape
import keras
from keras.layers import Input, Add, Dense, Activation, GRU, BatchNormalization, Flatten, Conv1D, \
    MaxPooling1D, Reshape
from keras.models import Model

import imblearn
import utils.tools as utils

def read_fasta(file):
    f = open(file)
    documents = f.readlines()
    string = ""
    flag = 0
    fea = []
    for document in documents:
        if document.startswith(">") and flag == 0:
            # if document.endswith(">") and flag == 0:
            flag = 1
            continue
        elif document.startswith(">") and flag == 1:
            # elif document.endswith(">") and flag == 1:
            string = string.upper()
            fea.append(string)
            string = ""
        else:
            string += document
            string = string.strip()
            string = string.replace(" ", "")

    fea.append(string)
    f.close()
    return fea

def makeplot(fpr, tpr, roc_auc, file_path):
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='DL_1 ROC (area = %.2f%%)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()

def calculate_seq_weight(fasta_list: list, pos_sample_len: int) -> list:
    """
    Return frequency list of 5-base sequence of each long sequence from fasta list.
    Steps:
    1. Extract 5-base sequence of each long sequence from fasta list.
    2. separate 5-base sequence into positive and negative samples.
    3. Calculate the count of each type of 5-base sequence of two samples.
    4. Calculate the frequency of each type of 5-base sequence of two samples.
    5. Create frequency dataframes with 5-base sequence, count, frequency of two samples.
    6. Based on the 5-base sequence in frequency dataframe and long sequence from fasta lis, match long sequence with
    its 5-base sequence frequency of each positive and negative sample and combine them together.
    7. The return list looks like: [0.14329896907216494, 0.14329896907216494,..., 0.007] with same len of fasta list.
    """
    seq_list = []
    freq_list = []
    # 1. Extract 5-base sequence of each long sequence from fasta list.
    for seq in fasta_list:
        sequence = seq[18:23]
        seq_list.append(sequence)
    seq_len = len(seq_list)
    # 2. separate 5-base sequence into positive and negative samples.
    pos_sample_list = seq_list[0:pos_sample_len]
    neg_sample_list = seq_list[pos_sample_len:]
    # 3. Calculate the count of each type of 5-base sequence of two samples.
    pos_count = Counter(pos_sample_list)
    neg_count = Counter(neg_sample_list)
    # 4.1 Calculate the frequency of each type of 5-base sequence of positive sample.
    # 5.1 Create frequency dataframe with 5-base sequence, count, frequency of positive sample.
    pos_df = pd.DataFrame.from_dict(pos_count, orient='index').reset_index()
    pos_df = pos_df.rename(columns={'index': 'RNA_sequence', 0: 'count'})
    pos_df["freq"] = pos_df["count"] / pos_sample_len
    # 4.2 Calculate the frequency of each type of 5-base sequence of negative sample.
    # 5.2 Create frequency dataframe with 5-base sequence, count, frequency of positive sample.
    neg_df = pd.DataFrame.from_dict(neg_count, orient='index').reset_index()
    neg_df = neg_df.rename(columns={'index': 'RNA_sequence', 0: 'count'})
    neg_df["freq"] = neg_df["count"] / (seq_len - pos_sample_len)
    # 6. Based on the 5-base sequence in frequency dataframe and long sequence from fasta lis, match long sequence with
    #    its 5-base sequence frequency of each positive and negative sample and combine them together.
    for i in range(len(seq_list)):
        # positive sample
        if i in range(0, pos_sample_len):
            if pos_df["RNA_sequence"].str.contains(seq_list[i]).any():
                location = pos_df.loc[pos_df["RNA_sequence"] == seq_list[i]].index[0]
                freq_list.append(pos_df.iloc[location]["freq"])
            else:
                print("sequence not in positive frequency list in %d row" % i)
        # negative sample
        elif i in range(pos_sample_len, seq_len):
            if neg_df["RNA_sequence"].str.contains(seq_list[i]).any():
                location = neg_df.loc[neg_df["RNA_sequence"] == seq_list[i]].index[0]
                freq_list.append(neg_df.iloc[location]["freq"])
            else:
                print("sequence not in negative frequency list in %d row" % i)
        else:
            print("sequence location exceeds the sequence list length, may due to unmatch of len(seq_list) and seq_len")
    # 7. The return list looks like: [0.14329896907216494, 0.14329896907216494,..., 0.007] with same len of fasta list.
    return freq_list

def CG(x):
    x1 = Conv1D(64, kernel_size=1, strides=1, padding='same', activation='relu')(x)
    x1 = BatchNormalization()(x1)
    x1 = Conv1D(32, kernel_size=1, strides=1, padding='same', activation='relu')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Flatten()(x1)
    x1 = Dense(32, activation='relu')(x1)
    x2 = GRU(32)(x)
    x = Add()([x1, x2])
    x = Activation('relu')(x)
    x = Reshape((32, 1))(x)
    return x

def rna_train(feature_list: list, deep_fusion: list, params: list, \
            random_seed: list, train_data: str, summary_df: pd.DataFrame, config: dict, logger: logging):
    
    wkDir = os.getcwd()
    dataset = train_data
    rna_decorate_method = config['train']['rna_decorate_method']
    # 根据路径2数据进行shape提取
    train_shape = summary_df.shape[1]

    # 对路径1-模型搭建路径的模型进行判断
    if rna_decorate_method == 'RESG':  # or if structPath1 == '':
        # 当structurePath1为空，或structureName为RESG模型
        # 对RESG模型进行搭建
        input = Input(shape=(train_shape, 1))
        x = Conv1D(128, kernel_size=3, strides=1, padding='same', activation='relu')(input)
        x = MaxPooling1D()(x)
        x = CG(x)
        x = CG(x)
        x = CG(x)
        x = Conv1D(64, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        x = MaxPooling1D()(x)
        x = Flatten()(x)

        x = Dense(32, activation='relu')(x)
        x = Dense(16, activation='relu')(x)
        x = Dense(2, activation='sigmoid')(x)
        model = Model(inputs=input, outputs=x)
    else:
        # 当structurePath1不为空，或structureName不为RESG模型(CNN模型)
        # 读取路径1-模型搭建路径中的模型路径
        # model = keras.models.load_model(structPath1, compile=False)
        # TODO: currently only support RESG
        raise RuntimeError('Currently only support RESG')

    positive = int(config['train']["positive_num"])
    negative = int(config['train']["negtive_num"])
    skfs = int(config['train']["cross_compare"])
    epochs = int(config['train']["epochs"])
    shuffle = config['train']["shuffle"]
    balanced_method = config['train']['imbalance']

    data_ = summary_df
    data = np.array(data_)
    data = data[:, 0:]
    [m1, n1] = np.shape(data)
    label1 = np.ones((positive, 1))  # Value can be changed
    label2 = np.zeros((negative, 1))
    label = np.append(label1, label2)

    # 权重选项
    if balanced_method == "Sample_weight":
        # read fasta file for calculate weight.
        fasta_seq_list = read_fasta(dataset)
        # calculate weight
        weight_sample = calculate_seq_weight(fasta_seq_list, positive)
        # change weight list to numpy array for model.
        weight_sample = np.array(weight_sample)
        # 模型训练（CNN模型与RESG模型是否需要分类）
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', weighted_metrics=[])
    elif balanced_method == "ADASYN":
        ada = imblearn.over_sampling.ADASYN(random_state=0)
        data_resample, label_resample = ada.fit_resample(data, label)
        logger.info("过采样ada后样本阴性阳性样本情况 %s" % Counter(label_resample))
        data = data_resample
        label = label_resample
        [m1, n1] = np.shape(data)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    else:
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    shu = scale(data)
    X = shu
    y = label
    sepscores = []
    sepscores_ = []

    ytest = np.ones((1, 2)) * 0.5
    yscore = np.ones((1, 2)) * 0.5
    ytrain = np.ones((1, 2)) * 0.5
    ytscore = np.ones((1, 2)) * 0.5

    X = X.reshape(m1, n1, 1)

    # shuffle开关
    if shuffle:
        skf = StratifiedKFold(shuffle=shuffle, random_state=666, n_splits=skfs)
    else:
        skf = StratifiedKFold(shuffle=shuffle, random_state=None, n_splits=skfs)

    index = 0
    data_list = []
    roc_value = []

    for train, test in skf.split(X, y):
        index += 1
        model_name = "model_%s" % (index)
        y_train = utils.to_categorical(y[train])  # generate the resonable results
        cv_clf = model
        # 权重选择2
        if balanced_method == "Sample_weight":
            hist = cv_clf.fit(X[train],
                                y_train,
                                sample_weight=np.array(weight_sample[train]),
                                epochs=epochs,
                                validation_data=(X[train], y_train, np.array(weight_sample[train])))
        elif balanced_method == "ADASYN":
            hist = cv_clf.fit(X[train],
                                y_train,
                                epochs=epochs)
        else:
            print("No use any weight method")
            hist = cv_clf.fit(X[train],
                                y_train,
                                epochs=epochs)
        # 训练集评估
        ytrain = np.vstack((ytrain, y_train))
        y_train_tmp_ = y[train]
        y_score_ = cv_clf.predict(X[train])  # the output of  probability
        ytscore = np.vstack((ytscore, y_score_))
        # error test
        fpr_, tpr_, _ = roc_curve(y_train[:, 0], y_score_[:, 0])

        roc_auc_ = auc(fpr_, tpr_)
        roc_value.append(roc_auc_)
        y_class_ = utils.categorical_probas_to_classes(y_score_)
        acc_, precision_, npv_, sensitivity_, specificity_, mcc_, f1_ = utils.calculate_performace(len(y_class_),
                                                                                                    y_class_,
                                                                                                    y_train_tmp_)
        sepscores_.append([acc_, precision_, npv_, sensitivity_, specificity_, mcc_, f1_, roc_auc_])

        model_path = os.path.join(wkDir, model_name + ".h5")
        cv_clf.save(model_path)
        auc_score_ = roc_auc_ * 100
        train_path = os.path.join(wkDir, "train_%s.png" % (index))
        makeplot(fpr_, tpr_, auc_score_, train_path)
        # 测试集评估
        y_test = utils.to_categorical(y[test])  # generate the test
        ytest = np.vstack((ytest, y_test))
        y_test_tmp = y[test]
        y_score = cv_clf.predict(X[test])  # the output of  probability
        yscore = np.vstack((yscore, y_score))
        fpr, tpr, _ = roc_curve(y_test[:, 0], y_score[:, 0])
        roc_auc = auc(fpr, tpr)
        y_class = utils.categorical_probas_to_classes(y_score)
        acc, precision, npv, sensitivity, specificity, mcc, f1 = utils.calculate_performace(len(y_class), y_class,
                                                                                            y_test_tmp)
        sepscores.append([acc, precision, npv, sensitivity, specificity, mcc, f1, roc_auc])
        auc_score = roc_auc * 100
        test_path = os.path.join(wkDir, "test_%s.png" % (index))
        makeplot(fpr, tpr, auc_score, test_path)
        data_temp = {
            "structure_name": model_name,
            "structure_path": model_path,
            "acc": acc_,
            "precision": precision_,
            "npv": npv_,
            "sensitivity": sensitivity_,
            "specificity": specificity_,
            "mcc": mcc_,
            "f1": f1_,
            "roc_auc": roc_auc_,
            "pltObj": [["train_%s" % (index), train_path, "resyn_2D"]],
            'multimedia': [
                {
                    "name": "evaluate",
                    "path": model_path,
                    "type": "",
                    "dataset": dataset,
                    "acc": acc,
                    "precision": precision,
                    "npv": npv,
                    "sensitivity": sensitivity,
                    "specificity": specificity,
                    "mcc": mcc,
                    "f1": f1,
                    "roc_auc": roc_auc,
                    "png": [["test_%s" % (index), test_path, "resyn_2D"]],
                    "feature_list": feature_list,
                    "deep_fusion": deep_fusion,
                    "random_seed": random_seed,
                    "params": params
                }],
            "status": "Success",
            "error": "",
            "isStruct": "False",
            "isResult": "True"
        }
        data_list.append(data_temp)
    # 训练集
    scores_ = np.array(sepscores_)
    row = ytscore.shape[0]
    ytscore = ytscore[np.array(range(1, row)), :]
    ytrain = ytrain[np.array(range(1, row)), :]
    fpr_, tpr_, _ = roc_curve(ytrain[:, 0], ytscore[:, 0])
    mean_train_score = np.mean(scores_, axis=0)
    auc_score = np.mean(scores_, axis=0)[7]
    auc_score = auc_score * 100
    mean_train_path = os.path.join(wkDir, "mean_train.png")
    makeplot(fpr_, tpr_, auc_score, mean_train_path)

    # get best model
    _model_best_index = roc_value.index(max(roc_value))
    _model_path = data_list[_model_best_index]['structure_path']
    Path(_model_path).rename('model.h5')

    # 测试集
    scores = np.array(sepscores)
    row = yscore.shape[0]
    yscore = yscore[np.array(range(1, row)), :]
    ytest = ytest[np.array(range(1, row)), :]
    fpr, tpr, _ = roc_curve(ytest[:, 0], yscore[:, 0])
    mean_test_score = np.mean(scores, axis=0)
    auc_score = np.mean(scores, axis=0)[7]
    auc_score = auc_score * 100
    mean_test_path = os.path.join(wkDir, "mean_test.png")
    makeplot(fpr, tpr, auc_score, mean_test_path)
    data_list.append({
        "structure_name": "平均计算",
        "structure_path": "",
        "acc": mean_train_score[0],
        "precision": mean_train_score[1],
        "npv": mean_train_score[2],
        "sensitivity": mean_train_score[3],
        "specificity": mean_train_score[4],
        "mcc": mean_train_score[5],
        "f1": mean_train_score[6],
        "roc_auc": mean_train_score[7],
        "pltObj": [["mean_train", mean_train_path, "resyn_2D"]],
        'multimedia': [
            {"name": "evaluate",
                "path": "",
                "type": "",
                "acc": mean_test_score[0],
                "precision": mean_test_score[1],
                "npv": mean_test_score[2],
                "sensitivity": mean_test_score[3],
                "specificity": mean_test_score[4],
                "mcc": mean_test_score[5],
                "f1": mean_test_score[6],
                "roc_auc": mean_test_score[7],
                "png": [["mean_test", mean_test_path, "resyn_2D"]]
                }],
        "status": "Success",
        "error": "",
        "isStruct": "False",
        "isResult": "True"
    })
    datadf = pd.DataFrame(data_list)
    return datadf

