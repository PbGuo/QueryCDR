import argparse
import os
from random import shuffle

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='data/', type=str,
                    help='The folder path')

parser.add_argument('--is_shuffled', default='0', type=int,
                    help='Needed to be shuffled')

def write(folder):
    # get the list of directories and separate them into 2 types: training and validation
    training_dirs = os.listdir(folder + "/train")
    testing_dirs = os.listdir(folder + "/test")

    # make 2 lists to save file paths
    training_names = []
    testing_names = []

    # append all files into 2 lists
    for training_item in training_dirs:
        train_flow_item = folder + "/train" + "/" + training_item
        training_names.append(train_flow_item)
    for testing_item in testing_dirs:
        test_flow_item = folder + "/test" + "/" + testing_item
        testing_names.append(test_flow_item)



    # print all file paths
    for i in training_names:
        print(i)
    for i in testing_names:
        print(i)


    # shuffle file names if set
    if args.is_shuffled == 1:
        shuffle(training_names)
        shuffle(testing_names)

    train_name = '../fine_flist/train'+folder.split('/')[0].split('_')[1]+folder.split('/')[1]+'.flist'
    test_name = '../fine_flist/test'+folder.split('/')[0].split('_')[1]+folder.split('/')[1]+'.flist'


    fo = open(train_name, "w")
    fo.write("\n".join(training_names))
    fo.close()

    fo = open(test_name, "w")
    fo.write("\n".join(testing_names))
    fo.close()

    # print process
    print("Written file is: ", train_name, ", is_shuffle: ", args.is_shuffled)


if __name__ == "__main__":

    args = parser.parse_args()

    write('dataset_1/data')
    write('dataset_1/gt')
    write('dataset_02/data')
    write('dataset_04/data')
    write('dataset_06/data')
    write('dataset_08/data')
    write('dataset_2/data')
    write('dataset_4/data')
    write('dataset_6/data')
    write('dataset_8/data')
    write('dataset_02/gt')
    write('dataset_04/gt')
    write('dataset_06/gt')
    write('dataset_08/gt')
    write('dataset_2/gt')
    write('dataset_4/gt')
    write('dataset_6/gt')
    write('dataset_8/gt')


    