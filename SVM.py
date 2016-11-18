"""
Method: SVM
Editor: Thanh L.X.
"""


from data import processData
from sklearn import svm
import copy as cp
import pickle


def to_vector(input_set, set_dict, base_vector, mode_train=False, labeled=True):
    vector = []
    labels = []
    length_dict = len(base_vector)
    if mode_train:
        for line in input_set:
            line = line.strip()
            label = line.split("\t")[0]
            content = line.split("\t")[1].split(",")
            vec = cp.copy(base_vector)
            for word in content:
                vec[set_dict.index(word)] += 1
            vector.append(vec)
            labels.append(label)
    else:
        if not labeled:
            for line in input_set:
                line = line.strip()
                content = line.split(",")
                vec = cp.copy(base_vector)
                for word in content:
                    if word in set_dict:
                        vec[set_dict.index(word)] += 1
                    else:
                        vec[length_dict-1] += 1
                vector.append(vec)
            return vector
        else:
            for line in input_set:
                line = line.strip()
                label = line.split("\t")[0]
                content = line.split("\t")[1].split(",")
                vec = cp.copy(base_vector)
                for word in content:
                    if word in set_dict:
                        vec[set_dict.index(word)] += 1
                    else:
                        vec[length_dict-1] += 1
                vector.append(vec)
                labels.append(label)
    return vector, labels


def evaluate(predict_label, test_label):
    assert len(predict_label) == len(test_label), '==predictions and test labels must have the same length=='
    length = len(predict_label)
    true_pred = 0.
    for go in range(length):
        if predict_label[go] == test_label[go]:
            true_pred += 1
    return true_pred/length


if __name__ == "__main__":

    print "=============================="
    print "Training-validating"
    split_ratio = 0.9
    source_path = "./data/"
    problem_id = "small"  # or "large" for 2nd problem
    source_train = "training-data-"+problem_id+".txt"
    source_test = "test-data-"+problem_id+".txt"
    train_set, valid_set = processData.split_data(source_path+source_train, split_ratio)
    set_dict = processData.to_dict(train_set)
    length_dict = len(set_dict)
    base_vector = [0] * (length_dict + 1)
    support_vector, support_label = to_vector(train_set, set_dict, base_vector, mode_train=True)
    test_vector, test_label = to_vector(valid_set, set_dict, base_vector, mode_train=False)
    clf = svm.SVC()
    clf.fit(support_vector, support_label)
    prediction = clf.predict(test_vector)
    accuracy = evaluate(prediction, test_label)
    print "Prediction result on valid-set: {}".format(prediction)
    print "Prediction accuracy on valid-set: {}".format(accuracy)

    print "=============================="
    print "Generate prediction on test-set:"
    train_set = processData.split_data(source_path + source_train, False)
    test_set = processData.split_data(source_path + source_test, False)
    set_dict = processData.to_dict(train_set)
    length_dict = len(set_dict)
    base_vector = [0] * (length_dict + 1)
    support_vector, support_label = to_vector(train_set, set_dict, base_vector, mode_train=True)
    test_vector = to_vector(test_set, set_dict, base_vector, mode_train=False, labeled=False)
    clf = svm.SVC()
    clf.fit(support_vector, support_label)
    prediction = clf.predict(test_vector)
    print "Prediction result on test-set: {}".format(prediction)
    # save to file
    with open(source_path+problem_id+"_svm_prediction_result", "wb") as handle:
        pickle.dump(prediction, handle)
    print "Done.."
