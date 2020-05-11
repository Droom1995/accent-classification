# load and evaluate a saved model
from numpy import loadtxt
from keras.models import load_model
import pandas as pd
import sys
sys.path.append('../speech-accent-recognition/src>')
import getsplit
import trainmodel

if __name__ == '__main__':
    '''
        Console command example:
        python evaluate.py bio_testdata.csv
    '''

    # Load arguments
    # print(sys.argv)
    file_name = sys.argv[1]
    # load model
    model = load_model('../models/model1.h5')
    # summarize model.
    model.summary()
    # load dataset
    df = pd.read_csv(file_name)

    # Filter metadata to retrieve only files desired
    filtered_df = getsplit.filter_df(df)
    X_train, X_test, y_train, y_test = getsplit.split_people(filtered_df)

    X_test = trainmodel.to_categorical(y_train)
    y_test = trainmodel.to_categorical(y_test)

    print(X_test)
    # evaluate the model
    score = model.evaluate(X_test, y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))
