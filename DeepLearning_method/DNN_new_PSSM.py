# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import math
from sklearn.metrics import auc


def readcsvfile(filename): #读文件，feature和tabel合并在一起
    list1=[] #features
    list2=[] #labels
    with open(filename) as f:
        for line in f:
            sl=line.split(',')
            example=[]
            if int(sl[-1])==0:
                label=[0,1]
            else:
                label=[1,0]
            for i in range(len(sl)-1):
                example.append((sl[i]))
            list1.append(example)
            list2.append(label)
    return list1,list2

#每个批次的大小
batch_size=40
#计算一共有多少个批次

#参数概要
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean=tf.reduce_mean(var)
        tf.summary.scalar('mean',mean)
    with tf.name_scope('stddev'):
        stddev=tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev',stddev)
    tf.summary.scalar('max',tf.reduce_max(var))
    tf.summary.scalar('min',tf.reduce_min(var))
    tf.summary.histogram('histogram',var)

with tf.name_scope('input'):
    #定义2个placeholder代表实际值
    x=tf.placeholder(tf.float32,[None,30])
    y=tf.placeholder(tf.float32,[None,2])
    keep_prob=tf.placeholder(tf.float32)

#创建一个简单的softmax神经网络
lr=tf.Variable(0.001,dtype=tf.float32)
with tf.name_scope('layer'):
    with tf.name_scope('w1'):
        W1=tf.Variable(tf.truncated_normal([30,20],stddev=0.1))
        variable_summaries(W1)
# W1=tf.Variable(tf.zeros([651,20]))
    with tf.name_scope('b1'):
        b1=tf.Variable(tf.zeros([20])+0.1)
        variable_summaries(b1)
    with tf.name_scope('tanh'):
        L1=tf.nn.tanh(tf.matmul(x,W1)+b1)
    with tf.name_scope('L1_drop'):
        L1_drop=tf.nn.dropout(L1,keep_prob)

    with tf.name_scope('w2'):
        W2=tf.Variable(tf.truncated_normal([20,20],stddev=0.1))
        variable_summaries(W2)
# W2=tf.Variable(tf.zeros([20,20]))
    with tf.name_scope('b2'):
        b2=tf.Variable(tf.zeros([20])+0.1)
        variable_summaries(b2)
    with tf.name_scope('tanh'):
        L2=tf.nn.tanh(tf.matmul(L1_drop,W2)+b2)
    with tf.name_scope('L2_drop'):
        L2_drop=tf.nn.dropout(L2,keep_prob)

    with tf.name_scope('w3'):
        W3=tf.Variable(tf.truncated_normal([20,20],stddev=0.1))
        variable_summaries(W3)
    with tf.name_scope('b3'):
        b3=tf.Variable(tf.zeros([20])+0.1)
        variable_summaries(b3)
    with tf.name_scope('tanh'):
        L3=tf.nn.tanh(tf.matmul(L2_drop,W3)+b3)
    with tf.name_scope('L3_drop'):
        L3_drop=tf.nn.dropout(L3,keep_prob)

    with tf.name_scope('w4'):
        W4= tf.Variable(tf.truncated_normal([20, 10], stddev=0.1))
        variable_summaries(W4)
    with tf.name_scope('b4'):
        b4= tf.Variable(tf.zeros([10]) + 0.1)
        variable_summaries(b4)
    with tf.name_scope('tanh'):
        L4 = tf.nn.tanh(tf.matmul(L3_drop, W4) + b4)
    with tf.name_scope('L4_drop'):
        L4_drop = tf.nn.dropout(L4, keep_prob)

    with tf.name_scope('w5'):
        W5= tf.Variable(tf.truncated_normal([10, 10], stddev=0.1))
        variable_summaries(W5)
    with tf.name_scope('b5'):
        b5= tf.Variable(tf.zeros([10]) + 0.1)
        variable_summaries(b5)
    with tf.name_scope('tanh'):
        L5 = tf.nn.tanh(tf.matmul(L4_drop, W5) + b5)
    with tf.name_scope('L5_drop'):
        L5_drop = tf.nn.dropout(L5, keep_prob)

    with tf.name_scope('w6'):
        W6=tf.Variable(tf.truncated_normal([10,2],stddev=0.1))
        variable_summaries(W6)
    with tf.name_scope('b6'):
        b6=tf.Variable(tf.zeros([2])+0.1)
        variable_summaries(b6)
    with tf.name_scope('softmax'):
        prediction=tf.nn.softmax(tf.matmul(L5_drop,W6)+b6)

    # with tf.name_scope('w5'):
    #     W5=tf.Variable(tf.truncated_normal([20,20],stddev=0.1))
    #     variable_summaries(W5)
    # with tf.name_scope('b5'):
    #     b5=tf.Variable(tf.zeros([20])+0.1)
    #     variable_summaries(b5)
    # with tf.name_scope('tanh'):
    #     L5=tf.nn.tanh(tf.matmul(L4_drop,W5)+b5)
    # with tf.name_scope('L5_drop'):
    #     L5_drop=tf.nn.dropout(L5,keep_prob)

    # with tf.name_scope('w6'):
    #     W6=tf.Variable(tf.truncated_normal([20,20],stddev=0.1))
    #     variable_summaries(W6)
    # with tf.name_scope('b6'):
    #     b6=tf.Variable(tf.zeros([20])+0.1)
    #     variable_summaries(b6)
    # with tf.name_scope('tanh'):
    #     L6=tf.nn.tanh(tf.matmul(L5_drop,W6)+b6)
    # with tf.name_scope('L6_drop'):
    #     L6_drop=tf.nn.dropout(L6,keep_prob)
    #
    # with tf.name_scope('w7'):
    #     W7=tf.Variable(tf.truncated_normal([20,20],stddev=0.1))
    #     variable_summaries(W7)
    # with tf.name_scope('b7'):
    #     b7=tf.Variable(tf.zeros([20])+0.1)
    #     variable_summaries(b7)
    # with tf.name_scope('tanh'):
    #     L7=tf.nn.tanh(tf.matmul(L6_drop,W7)+b7)
    # with tf.name_scope('L7_drop'):
    #     L7_drop=tf.nn.dropout(L7,keep_prob)
    #
    # with tf.name_scope('w6'):
    #     W6=tf.Variable(tf.truncated_normal([20,2],stddev=0.1))
    #     variable_summaries(W6)
    # with tf.name_scope('b6'):
    #     b6=tf.Variable(tf.zeros([2])+0.1)
    #     variable_summaries(b6)
    # with tf.name_scope('softmax'):
    #     prediction=tf.nn.softmax(tf.matmul(L5_drop,W6)+b6)

with tf.name_scope('loss'):
    #交叉熵代价函数
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    train_step=tf.train.AdamOptimizer(lr).minimize(loss)

#结果存放在一个布尔型列表中
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

def ROC(prob,test_label,drawROC,Is_ten_cross):
    num=len(prob)
    p=[]
    t=[]
    for i in range(num):
        p.append(prob[i][0])
        t.append(test_label[i][0])
    data = pd.DataFrame(index=range(0, num), columns=('probability', 'The true label'))
    data['The true label'] = t
    data['probability'] = p
    data.sort_values('probability', inplace=True, ascending=False)
    TPRandFPR = pd.DataFrame(index=range(len(data)), columns=('TP', 'FP'))

    for j in range(len(data)):
        data1 = data.head(n=j + 1)
        FP = len(data1[data1['The true label'] == 0][data1['probability'] >= data1.head(len(data1))['probability']])/float(len(data[data['The true label'] == 0]))
        TP = len(data1[data1['The true label'] == 1][data1['probability'] >= data1.head(len(data1))['probability']])/float(len(data[data['The true label'] == 1]))
        TPRandFPR.iloc[j] = [TP, FP]
    AUC = auc(TPRandFPR['FP'], TPRandFPR['TP'])
    if drawROC >= 1:
        plt.scatter(x=TPRandFPR['FP'], y=TPRandFPR['TP'],s=5, label='(FPR,TPR)', color='k')
        plt.plot(TPRandFPR['FP'], TPRandFPR['TP'], 'k', label='AUC = %0.5f' % AUC)
        plt.legend(loc='lower right')
        plt.title('Receiver Operating Characteristic')
        plt.plot([(0, 0), (1, 1)], 'r--')
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 01.01])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        # plt.show()
        if Is_ten_cross==0:
            plt.savefig("./Picture/DNN_ROC"+str(drawROC)+".png")
        else:
            plt.savefig("./Picture/DNN_ROC_"+str(drawROC)+"_cross.png")
        plt.clf()
    return AUC,TPRandFPR['TP'],TPRandFPR['FP']

def JudgePositive(test_prob,th):
    IsPositive=[]
    for i in range(len(test_prob)):
        if test_prob[i][0]>=th:
            IsPositive.append(1)
        else:
            IsPositive.append(0)
    return IsPositive


def Save_csv(data,name):
    csvfile = open('./Save_csv/' + name, 'w', newline='')
    writer = csv.writer(csvfile)
    writer.writerow(['TPR', 'FPR'])
    for i in range(len(data[0])):
        writer.writerow([data[0][i], data[1][i]])
    csvfile.close()

def Save_evaluation(data,name):
    csvfile = open('./evaluation/'+ name,'w',newline='')
    writer = csv.writer(csvfile)
    writer.writerow(['Accuracy', 'Specitivity','Sensitivity','AUC'])
    for i in range(len(data)):
        writer.writerow(data[i])
    csvfile.close()


def DNN(train_file,test_file,Isrestore):
    # 合并所有的summary
    merged = tf.summary.merge_all()
    MaxACC, MaxAUC=0,0
    target=[]
    resultACC=[]
    resultAUC=[]
    (train_feature, train_label) = readcsvfile(train_file)
    (test_feature, test_label) = readcsvfile(test_file)
    n_batch = len(train_feature) // batch_size
    saver=tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('logs/', sess.graph)
        for epoch in range(1):
            if Isrestore == 0:
                sess.run(tf.assign(lr, 0.001 * (0.95 ** epoch)))
                TP, TN, FP, FN = 0, 0, 0, 0
                for batch in range(n_batch):
                    batch_xs = train_feature[batch * batch_size:(batch + 1) * batch_size]
                    batch_ys = train_label[batch * batch_size:(batch + 1) * batch_size]
                    sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})  # 训练0.7
                    summary= sess.run(merged, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
            else:
                saver.restore(sess, './Model/DNN_model_' + str(Isrestore) + '.ckpt')
            test_acc = sess.run(accuracy, feed_dict={x: test_feature, y: test_label, keep_prob: 1.0})
            train_acc = sess.run(accuracy, feed_dict={x: train_feature, y: train_label, keep_prob: 1.0})
            # print("Iter " + str(epoch+1) + ", Test Accuracy " + str(test_acc) + ", Train Accuracy " + str(train_acc))
            test_prob = sess.run(prediction, feed_dict={x: test_feature, y: test_label, keep_prob: 1.0})
            test_pre = JudgePositive(test_prob, 0.6)
            writer.add_summary(summary, epoch)  # 将summary写入到logs中
            for c in range(len(test_pre)):
                if test_pre[c] == test_label[c][0]:
                    if test_pre[c] == 1:
                        TP += 1
                    else:
                        TN += 1
                else:
                    if test_pre[c] == 1:
                        FP += 1
                    else:
                        FN += 1
            Accuracy = (TP + TN) / (TP + TN + FN + FP)
            Specitivity = TN / (TN + FP)
            Sensitivity = TP / (TP + FN)
            AUC,TPR,FPR = ROC(test_prob, test_label, epoch + 1,0)
            if Isrestore == 0:
                target.append([Accuracy,Specitivity,Sensitivity,AUC])
                Save_csv([TPR,FPR],'DNN_iter'+str(epoch+1)+'.csv')
            # print("number:"+str(len(test_pre))+"\nAccuracy:" + str(Accuracy) + "\nSpecitivity:" + str(
            #     Specitivity) + "\nSensitivity:" + str(Sensitivity) + "\nAUC:" + str(AUC))
            if Isrestore == 0:
                if Accuracy>MaxACC:
                    MaxACC=Accuracy
                    resultACC = [epoch+1,len(test_pre),Accuracy,Specitivity,Sensitivity,AUC]
                if AUC>MaxAUC:
                    MaxAUC=AUC
                    resultAUC = [epoch + 1,len(test_pre), Accuracy, Specitivity, Sensitivity, AUC]
                saver.save(sess, './Model/DNN_model_' + str(epoch + 1) + '.ckpt')
            else:
                break
            # if epoch == resultAUC[0]-1:
            #     saver.save(sess, "./net/DNN.model")
        # print(test_prob)
        # print(test_pre)
        if Isrestore==0:
            Save_target(target)
            print("MaxACC: Iter: "+str(resultACC[0])+ "\nnumber:" + str(resultACC[1]) + "\nAccuracy:" + str(resultACC[2]) + "\nSpecitivity:" + str(
                resultACC[3]) + "\nSensitivity:" + str(resultACC[4]) + "\nAUC:" + str(resultACC[5]))
            print("MaxAUC: Iter: "+str(resultAUC[0])+"\nnumber:" + str(resultAUC[1]) + "\nAccuracy:" + str(resultAUC[2]) + "\nSpecitivity:" + str(
                resultAUC[3]) + "\nSensitivity:" + str(resultAUC[4]) + "\nAUC:" + str(resultAUC[5]))

def K_cross(train_file,test_file,k):
    cross_MaxACC,cross_MaxAUC = 0,0
    cross_resultAUC = []
    cross_resultACC = []
    cross_evaluation=[]
    MaxACC, MaxAUC = 0, 0
    resultAUC = []
    resultACC = []
    evaluation=[]

    (data_feature, data_label) = readcsvfile(train_file)
    (test_feature, test_label) = readcsvfile(test_file)
    n=len(data_feature)//k
    n_batch = ((k-1)*n) // batch_size

    with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(1):
                sess.run(tf.assign(lr, 0.001 * (0.95 ** epoch)))
                test_pre_combine = []
                test_prob_combine = []
                TP, TN, FP, FN = 0, 0, 0, 0
                for i in range(k):
                    test_feature = data_feature[n * i:n * (i + 1)]
                    test_label = data_label[n * i:n * (i + 1)]
                    train_feature = data_feature[0:n * i] + data_feature[n * (i + 1):k * n]
                    train_label = data_label[0:n * i] + data_label[n * (i + 1):k * n]
                    for batch in range(n_batch):
                        batch_xs = train_feature[batch * batch_size:(batch + 1) * batch_size]
                        batch_ys = train_label[batch * batch_size:(batch + 1) * batch_size]
                        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})  # 训练0.
                    test_prob= sess.run(prediction, feed_dict={x: test_feature, y: test_label, keep_prob: 1.0})
                    test_pre = JudgePositive(test_prob, 0.5)
                    test_pre_combine.extend(test_pre)
                    test_prob_combine.extend(test_prob)
                for c in range(len(test_pre_combine)):
                    if test_pre_combine[c] == data_label[c][0]:
                        if test_pre_combine[c] == 1:
                            TP += 1
                        else:
                            TN += 1
                    else:
                        if test_pre_combine[c] == 1:
                            FP += 1
                        else:
                            FN += 1
                Accuracy = (TP + TN) / float(TP + TN + FN + FP)
                Specificity = TN / float(TN + FP)
                Sensitivity = TP / float(TP + FN)
                AUC, TPR, FPR = ROC(test_prob_combine, data_label, epoch+1, 1)
                MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
                cross_evaluation.append([Accuracy, Specificity, Sensitivity, AUC,MCC])
                Save_evaluation(cross_evaluation, 'cross_evaluation' + str(epoch + 1) + '.csv')
                Save_csv([TPR, FPR], 'DNN_epoch' + str(epoch + 1) + '.csv')
                if Accuracy > cross_MaxACC:
                    cross_MaxACC = Accuracy
                    cross_resultACC = [epoch + 1, len(test_prob_combine), Accuracy, Specificity, Sensitivity, AUC]
                if AUC > cross_MaxAUC:
                    cross_MaxAUC = AUC
                    cross_resultAUC = [epoch + 1, len(test_prob_combine), Accuracy, Specificity, Sensitivity, AUC]

                    #independent dataset
                test_prob = sess.run(prediction, feed_dict={x: test_feature, y: test_label, keep_prob: 1.0})
                test_pre = JudgePositive(test_prob, 0.5)

                for c in range(len(test_pre)):
                    if test_pre[c] == test_label[c][0]:
                        if test_pre[c] == 1:
                            TP += 1
                        else:
                            TN += 1
                    else:
                        if test_pre[c] == 1:
                            FP += 1
                        else:
                            FN += 1
                Accuracy = (TP + TN) / (TP + TN + FN + FP)
                Specificity = TN / (TN + FP)
                Sensitivity = TP / (TP + FN)
                AUC, TPR, FPR = ROC(test_prob, test_label, epoch + 1, 0)
                MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
                evaluation.append([Accuracy, Specificity, Sensitivity, AUC,MCC])
                Save_evaluation(evaluation, 'independent_evaluation' + str(epoch + 1) + '.csv')  # 保存AUC,ACC,SP
                Save_csv([TPR, FPR], 'DNN_epoch_independent' + str(epoch + 1) + '.csv')
                if Accuracy > MaxACC:
                    MaxACC = Accuracy
                    resultACC = [epoch + 1, len(test_pre), Accuracy, Specificity, Sensitivity, AUC]

                    if AUC > MaxAUC:
                        MaxAUC = AUC
                        resultAUC = [epoch + 1, len(test_pre), Accuracy, Specificity, Sensitivity, AUC]

            print("ten_cross_MaxAUC: Iter: " + str(cross_resultAUC[0]) + "\nnumber:" + str(
                cross_resultAUC[1]) + "\nAccuracy:" + str(
                cross_resultAUC[2]) + "\nSpecificity:" + str(cross_resultAUC[3]) + "\nSensitivity:" + str(
                cross_resultAUC[4]) + "\nAUC:"
                  + str(cross_resultAUC[5]))
            print("MaxACC: Iter: " + str(resultAUC[0]) + "\nnumber:" + str(resultAUC[1]) + "\nAccuracy:" + str(
                resultAUC[2]) + "\nSpecificity:" + str(resultAUC[3]) + "\nSensitivity:" + str(
                resultAUC[4]) + "\nAUC:" + str(resultAUC[5]))


if __name__=='__main__':
    is_ten_cross=1

    if is_ten_cross==0:
        train='Homo_15_trainPSSM_random.csv'
        test='Homo_15_testPSSM_random.csv'
        DNN(train,test,0)

    else:
        train = 'Homo_15_trainPSSM_random.csv'
        test = 'Homo_15_testPSSM_random.csv'
        K_cross(train,test,4)



