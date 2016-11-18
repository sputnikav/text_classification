import seaborn as sns
import random
import numpy as np
# import copy as cp


def visualize(data, x_name, y_name, title_name):
    axis = sns.distplot(data)
    sns.set(color_codes=True)
    axis.set(xlabel=x_name, ylabel=y_name, title=title_name)
    sns.plt.show()


def get_data(input):
    list_dict = []
    list_length = []
    with open(input, "r") as infile:
        for line in infile:
            line = line.strip()
            if line:
                content = line.split("\t")[1]
                strin = content.split(",")
                list_length.append(len(strin))
                for word in strin:
                    list_dict.append(word)
    return list_dict, list_length


def to_dict(train_set):
    list_dict = []
    for line in train_set:
        line = line.strip()
        if line:
            # label = line.split()[0]
            content = line.split("\t")[1].split(",")
            for word in content:
                if word not in list_dict:
                    list_dict.append(word)
    return list_dict


def split_data(input, split_ratio):
    dataset = []
    with open(input) as infile:
        for line in infile:
            line = line.strip()
            dataset.append(line)
    if not split_ratio:
        return list(dataset)
    train_size = int(len(dataset) * split_ratio)
    train_set = []
    valid_set = list(dataset)
    while len(train_set) < train_size:
        index = random.randrange(len(valid_set))
        train_set.append(valid_set.pop(index))
    return [train_set, valid_set]


# load Dictionary
def one_hot_table(length_dict):

    # list_class = np.empty(length_dict)
    table_dict = np.zeros((length_dict, length_dict))
    embedding_dict = np.arange(length_dict)
    table_dict[embedding_dict, embedding_dict] = 1
    # print (Table_Dict)
    # print (Table_Dict.shape)
    return table_dict


def scan(test_set, set_dict):
    new_test_set = []
    for line in test_set:
        line = line.strip()
        strin = line.replace(",", " ")
        for word in line.split(","):
            if word not in set_dict:
                strin = strin.replace(word, " ")
        new_test_set.append(strin)
    return new_test_set


if __name__ == "__main__":

    # ================== #
    # - SMALL DATA SET - #
    # ================== #
    print "Train_small checkup"
    list_dict, list_length = get_data("training-data-small.txt")
    set_dict = set(list_dict)
    print (len(set_dict))
    visualize(list_length, x_name="length", y_name="estimatePDF", title_name="length-distribution"
                                                                             " of input-representation")
    # ================== #
    # - LARGE DATA SET - #
    # ================== #
    print "Train_large checkup"
    list_dict, list_length = get_data("training-data-large.txt")
    set_dict = set(list_dict)
    print (len(set_dict))
    visualize(list_length[:100000], x_name="length", y_name="estimatePDF", title_name="length-distribution"
                                                                                      " of input-representation")
