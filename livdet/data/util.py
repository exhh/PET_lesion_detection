import os, sys

def get_data_ids(traindata):
    traindata_tuple = ('liver3Dnpy', )
    if traindata in traindata_tuple:
        traindatasets = ('The folder names of training subjects', )
        validdatasets = ('The folder names of validation subjects', )
        testdatasets = ('The folder names of test subjects',)

    return traindatasets, validdatasets, testdatasets
