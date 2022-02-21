import os.path


def count():
    with open("./crosswoz/train_with_location.txt") as train:
        print_result(train, "train")
    with open("./crosswoz/dev_with_location.txt") as dev:
        print_result(dev, "dev")
    with open("./crosswoz/test_with_location.txt") as test:
        print_result(test, "test")


def print_result(file, filename):
    lines = file.readlines()
    print("slot of {}: {}".format(filename, count_slots(lines)))
    print("intent of {}: {}".format(filename, count_intent(lines)))
    print("sentence of {}: {}".format(filename, count_sentences(lines)))


def count_slots(lines):
    slot_list = []
    for line in lines:
        split_line = line.split(" ")
        if len(split_line) > 1:
            slot_list.append(split_line[1])
    return len(set(slot_list))


def count_intent(lines):
    intent_list = []
    for line in lines:
        split_line = line.split(" ")
        if len(split_line) == 1:
            intent_list.append(split_line[0])
    return len(set(intent_list))


def count_sentences(lines):
    result = 0
    for line in lines:
        split_line = line.split(" ")
        if len(split_line) == 1 and split_line[0] == '\n':
            result += 1
    return result


def count_dataset(dataset_name):
    intent_statistics = {}
    slot_statistics = {}
    if os.path.isfile("./{}/train.txt".format(dataset_name)):
        with open("./{}/train.txt".format(dataset_name), "r") as file:
            intent_statistics, slot_statistics, avg_len = process_file(intent_statistics, slot_statistics, file)

    if os.path.isfile("./{}/train.txt".format(dataset_name)):
        with open("./{}/train.txt".format(dataset_name), "r") as file:
            intent_statistics, slot_statistics, avg_len = process_file(intent_statistics, slot_statistics, file)

    if os.path.isfile("./{}/train.txt".format(dataset_name)):
        with open("./{}/train.txt".format(dataset_name), "r") as file:
            intent_statistics, slot_statistics, avg_len = process_file(intent_statistics, slot_statistics, file)

    print("intent\tnumber")
    for intent in intent_statistics.keys():
        print(intent + "\t" + str(intent_statistics[intent]))
    print()

    print("slot\tnumber")
    for slot in slot_statistics.keys():
        print(slot + "\t" + str(slot_statistics[slot]))

    print()
    print("avg_len:{}".format(avg_len))


def process_file(intent_statistics, slot_statistics, file):
    sentence_len = 0
    total_len = 0
    total_cnt = 0
    for line in file.readlines():
        line = line.split(" ")
        if len(line) == 3:
            dict_inc_item(slot_statistics, line[1])
            sentence_len += 1
        elif len(line) == 1 and line[0] != "\n":
            dict_inc_item(intent_statistics, line[0].replace("\n", ""))
            total_len += sentence_len
            total_cnt += 1
            sentence_len = 0
    return intent_statistics, slot_statistics, 1.0 * total_len/total_cnt


def dict_inc_item(dict_name, item):
    if item in dict_name.keys():
        dict_name[item] += 1
    else:
        dict_name[item] = 1


if __name__ == '__main__':
    count_dataset("SMP-ECDT")
