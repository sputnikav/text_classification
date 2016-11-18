"""
Method: Naive Bayes
Editor: Thanh L.X.
"""

from data import processData
from collections import defaultdict
import math
import pickle


def read_data_train(trainset):
    dict_words = defaultdict(lambda: defaultdict(int))
    count_0 = 0
    count_1 = 0
    for line in trainset:
        line = line.strip()
        if line:
            label = line.split("\t")[0]
            content = line.split("\t")[1]
            if label == "0":
                count_0 += 1
            else:
                count_1 += 1
            for word in content.split(","):
                dict_words[label][word] += 1
    return count_0, count_1, dict_words


def bayes_predict(test_set, count_0, count_1, dict_words, labeled=True):
    alpha = 0.1
    prob_class_0 = float(count_0) / (count_0 + count_1)
    prob_class_1 = 1 - prob_class_0
    count_word_0 = sum(vocab['0'].values())
    count_word_1 = sum(vocab['1'].values())
    length_dict = len(dict_words)
    true_detect = 0.
    prediction = []
    for line in test_set:
        line = line.strip()
        if line:
            if labeled:
                label = line.split("\t")[0]
                content = line.split("\t")[1].split(",")
            else:
                content = line.split(",")
            prob_0 = math.log(prob_class_0)
            prob_1 = math.log(prob_class_1)

            for word in content:
                # if dict_words["0"][word] < 1000 or dict_words["1"][word] < 1000:
                prob_0 += math.log((dict_words["0"][word] + alpha)/(count_word_0 + alpha * length_dict))
                prob_1 += math.log((dict_words["1"][word] + alpha)/(count_word_1 + alpha * length_dict))
            if prob_0 > prob_1:
                predict = "0"
            else:
                predict = "1"
            prediction.append(predict)
            if labeled:
                if predict == label:
                    true_detect += 1
    if labeled:
        return true_detect, prediction
    else:
        return prediction


if __name__ == "__main__":
    print "=============================="
    print "Training-validating"
    split_ratio = 0.9
    source_path = "./data/"
    problem_id = "small"  # or "large" for 2nd problem
    source_train = "training-data-" + problem_id + ".txt"
    source_test = "test-data-" + problem_id + ".txt"
    train_set, valid_set = processData.split_data(source_path+source_train, split_ratio)
    count_0, count_1, vocab = read_data_train(train_set)
    print "check number of samples in each class"
    print "Total samples - class 0: {}".format(count_0)
    print "Total samples - class 1: {}".format(count_1)
    true_predict, prediction = bayes_predict(valid_set, count_0, count_1, vocab, labeled=True)
    accuracy = true_predict/len(valid_set)
    print "Prediction result on valid-set: {}".format(prediction)
    print "Prediction accuracy on valid-set: {}".format(accuracy)

    print "=============================="
    print "Generate prediction on test-set:"
    train_set = processData.split_data(source_path + source_train, False)
    test_set = processData.split_data(source_path + source_test, False)
    count_0, count_1, vocab = read_data_train(train_set)
    print "Total samples - class 0: {}".format(count_0)
    print "Total samples - class 1: {}".format(count_1)
    prediction = bayes_predict(test_set, count_0, count_1, vocab, labeled=False)
    print "Prediction result on test-set: {}".format(prediction)
    # save to file
    with open(source_path+problem_id+"_bayes_prediction_result", "wb") as handle:
        pickle.dump(prediction, handle)
    print "Done.."
