#!/usr/bin/python
#
# Perform optical character recognition, usage:
#     python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors: (Akash Bhapkar (abhapkar), Anurag Hambir (ahambir) and Roopank Kohli (rookohli))
# (based on skeleton code by D. Crandall, Oct 2020)
#

from PIL import Image
import sys
import math

CHARACTER_WIDTH = 14
CHARACTER_HEIGHT = 25
TRAINING_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "


def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        # result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y
        # in range(0, CHARACTER_HEIGHT) ], ]
        result += [['*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg + CHARACTER_WIDTH) for y in
                    range(0, CHARACTER_HEIGHT)], ]

    return result


def load_training_letters(fname):
    TRAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return {TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS))}


def convert_Dict_To_Array(train_letters):
    return [['.' for _ in range(2)] for _ in range((len(train_letters)))]


# This method is used to calculate emission probability of a word. Weights are assigned in order to give more weightage
# to a hit rather than a miss
def calculateSimpleProbability(testChar, trainChar):
    weight = 0.8
    val = 1
    for i in range(len(testChar)):
        if testChar[i] != trainChar[i]:
            val = val * (1 - weight)
        else:
            val = val * weight
    return val


def computeSimpleProbability(test_letters, data):
    output = ""
    valDict = {}
    for test_letter in test_letters:
        for index in range(len(data)):
            probValue = calculateSimpleProbability(test_letter, data[index][1])
            valDict[data[index][0]] = probValue
        output += max(valDict, key=valDict.get)
    print("Simple: " + output)


# Method to read and filter training data (bc.train). As this file contains tags along with original text, those tags
# were ignored while training the classifier.
def readTrainingData(dataset):
    dataList = []
    file = open(dataset, 'r')
    for line in file:
        data = tuple([w for w in line.split()])

        dataList += [(data[0::2]), ]

    finalData = ""

    for line in dataList:
        cleanLine = ""
        for word in line:
            if len(word) > 1:
                cleanLine += " " + ''.join(char for char in word if char in TRAINING_LETTERS)
        cleanLine = cleanLine.replace(" .", ".").replace("  ", " ")

        finalData += cleanLine + "\n"

    return finalData.strip().splitlines()


def computeInitialProbability(trainData):
    freqDict = {}
    probDict = {}
    charCount = 0

    for line in trainData:
        if len(list(line)) > 1:
            if list(line)[0] == " ":
                ch = list(line)[1]
            else:
                ch = list(line)[0]

            cvalue = freqDict.get(ch, 0)
            freqDict[ch] = cvalue + 1

    total_initial = 0
    for key in freqDict:
        total_initial += freqDict[key]

    for key in freqDict:
        probDict[key] = -math.log(float(freqDict[key]) / total_initial)

    for line in trainData:
        for _ in line:
            charCount += 1

    return freqDict, probDict, charCount


def computeTransitionProbability(trainData, initial_freq_dict):
    freqDict = {}
    probDict = {}

    for line in trainData:
        for index in range(len(line) - 1):
            cvalue = freqDict.get(line[index] + line[index + 1], 0)
            freqDict[line[index] + line[index + 1]] = cvalue + 1

    for key in freqDict:
        try:

            value = initial_freq_dict.get(key[0])
            probDict[key] = -math.log(float(freqDict[key]) / value)
        except Exception:
            continue

    return probDict


def computeHmmViterbiProbability(total_char, initial_prob_dict, transition_prob):
    dictOne = {}
    dictTwo = {}
    output = ""
    count = 0
    bufferValue = -math.log(1.0 / total_char)

    flag = False
    min_dict = {}

    for test_letter in test_letters:
        flag = not flag
        if flag:
            dictOne = {}
        else:
            dictTwo = {}
        for letter in TRAINING_LETTERS:
            emission_prob = -math.log(calculateSimpleProbability(train_letters[letter], test_letter))
            if count == 0:
                dictOne[letter] = (initial_prob_dict.get(letter, bufferValue) + emission_prob)
            else:
                viterbi_dict = {}
                for l in TRAINING_LETTERS:

                    transition_prob_value = transition_prob.get(l + letter, bufferValue)

                    unique_dict_key = str(count) + str(l) + str(letter)
                    if flag:
                        viterbi_dict[unique_dict_key] = (
                                transition_prob_value + emission_prob + dictTwo.get(str(l),
                                                                                    bufferValue))

                    else:
                        viterbi_dict[unique_dict_key] = (
                                transition_prob_value + emission_prob + dictOne.get(str(l),
                                                                                    bufferValue))

                minKey = min(viterbi_dict, key=viterbi_dict.get)
                min_dict[str(count) + letter] = minKey[-2]
                if flag:
                    dictOne[letter] = viterbi_dict[minKey]
                else:
                    dictTwo[letter] = viterbi_dict[minKey]
        if flag:
            minimum = min(dictOne, key=dictOne.get)
        else:
            minimum = min(dictTwo, key=dictTwo.get)

        count += 1
        output += str(minimum)
    temp_count = count - 1

    final_output = minimum
    while temp_count > 0:
        minimum = min_dict[str(temp_count) + str(minimum)]
        temp_count -= 1
        final_output = minimum + final_output

    print("   HMM: " + output)


#####
# main program

if __name__ == "__main__":

    if len(sys.argv) != 4:
        raise Exception("Usage: python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png")

    (train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
    train_letters = load_training_letters(train_img_fname)
    test_letters = load_letters(test_img_fname)

    data = convert_Dict_To_Array(train_letters)

    # Assigning training letters and its corresponding array representation to a list.
    for count, value in enumerate(train_letters):
        data[count][0] = value
        data[count][1] = train_letters[value]

    computeSimpleProbability(test_letters, data)

    trainData = readTrainingData(train_txt_fname)

    initial_freq_dict, initial_prob_dict, total_char = computeInitialProbability(trainData)

    transition_prob = computeTransitionProbability(trainData, initial_freq_dict)

    computeHmmViterbiProbability(total_char, initial_prob_dict, transition_prob)
