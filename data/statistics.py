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

if __name__ == '__main__':
    count()