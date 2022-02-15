def transfer_manual_dataset():
    with open("./manual_data/train_cn-sub-slot.out") as out_file:
        with open("./manual_data/train_cn.in") as in_file:
            target_list = out_file.readlines()
            input_list = in_file.readlines()
            for i in range(len(target_list)):
                process_single(target_list[i], input_list[i])


def process_single(target, input_sentence):
    output_word = []

    sentence = input_sentence.replace(" ", "")
    input_words = input_sentence.split(" ")
    output_seq = target.split(" ")
    intent = output_seq[-1]
    slot_label = []

    for i, word in enumerate(input_words):
        if slot_label[i] != "O":
            output_word.append(word)
            if output_seq[i][0] == 'B':
                slot_label.append(output_seq[i])
                count = len(word) - 1
            elif output_seq[i][0] == 'I':
                count = len(word)
        else:
            output_word = output_word + list(word)

    lattice = []
    index = 0
    for i, word in enumerate(input_words):
        if slot_label[i] != "O":
            for j in range(word):
                lattice.append(index)
            index += len(word)
        else:
            lattice.append(index)
            index += 1


    result = ""
    for i in range(sentence):
        result += sentence[i] + " " + output_seq[i] + " " + output_word + " " +


