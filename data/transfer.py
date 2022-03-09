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
        result += sentence[i] + " " + output_seq[i] + " " + output_word + " "


def crosswoz_to_sfid():
    with open("crosswoz/test.txt", encoding="UTF-8") as file:
        with open("crosswoz/sfid/test/label", "w", encoding="UTF-8") as label_file:
            with open("crosswoz/sfid/test/seq.in", "w", encoding="UTF-8") as in_file:
                with open("crosswoz/sfid/test/seq.out", "w", encoding="UTF-8") as out_file:
                    process_file(file, label_file, in_file, out_file)
                    print("ok")


def crosswoz_to_MLWA():
    with open("crosswoz/dev.txt", encoding="UTF-8") as file:
        with open("crosswoz/dev_MLWA.txt", "w", encoding="UTF-8") as MLWA_file:
                    process_file_MLWA(file, MLWA_file)
                    print("ok")


def process_file_MLWA(file, MLWA_file):
    sentence = []
    out_seq = []
    word = []
    for line in file.readlines():
        line = line.split(" ")
        if len(line) == 3:
            sentence.append(line[0])
            out_seq.append(line[1])
            word.append(line[2])
        elif len(line) == 1 and line[0] != "\n":
            for i in range(len(sentence)):
                if len(word[i]) == 2:
                    MLWA_file.write(sentence[i] + " " + out_seq[i] + " " + "B" + "\n")
                elif i == 0 or len(word[i-1]) == 2:
                    MLWA_file.write(sentence[i] + " " + out_seq[i] + " " + "B" + "\n")
                elif i == len(sentence) - 1 or len(word[i+1]) == 2:
                    MLWA_file.write(sentence[i] + " " + out_seq[i] + " " + "E" + "\n")
                else:
                    MLWA_file.write(sentence[i] + " " + out_seq[i] + " " + "I" + "\n")
            MLWA_file.write(line[0] + "\n")
            sentence = []
            out_seq = []
            word = []




def process_file(file, label_file, in_file, out_file):
    sentence = ""
    out_seq = ""
    for line in file.readlines():
        line = line.split(" ")
        if len(line) == 3:
            sentence += line[0] + " "
            out_seq += line[1] + " "
        elif len(line) == 1 and line[0] != "\n":
            sentence = sentence[:-1]
            out_seq = out_seq[:-1]
            label_file.write(line[0])
            in_file.write(sentence + "\n")
            out_file.write(out_seq + "\n")
            sentence = ""
            out_seq = ""


if __name__ == '__main__':
    crosswoz_to_MLWA()
