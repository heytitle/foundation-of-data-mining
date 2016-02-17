import mnist_dataloader as data
import math
import time
import pandas

from instance import Instance

NUM_LABEL = 10

def run():
    ( train, undef, test ) =  data.load_data_wrapper()
    log( "Trainning set : %8d instances", ( len(train) ) )
    log( "Testing set   : %8d instances", ( len(test) ) )

    txt = "Converting %s set to Instance objects"
    log( txt, ("training") )
    train = [ Instance( t[0], label_array=t[1] ) for t in train ]

    log( txt, ("test") )
    test  = [ Instance( t[0], label=t[1] ) for t in test ]

    instance = 1
    start_compare = time.time()
    """ Iterate through testing set """
    test_subset = test[0:10]
    for i in test_subset:
        log("Instance %d", ( instance ))

        """ Find the closest pair from training set
        """
        closest_pair = train[0]
        max_dist     = cosine_dist( i, closest_pair )

        for j in train[1:1000]:
            dist = cosine_dist( i, j )
            if( dist < max_dist ):
                max_dist     = dist
                closest_pair = j

            if( dist == 0 ):
                break

        i.predicted_label = closest_pair.label

        log(">>> %d, actual : %s , predict : %s", ( instance, test[0].label, test[0].predicted_label) )
        instance+=1
    end_compare = time.time()

    """ Compute confusion_matrix, accuracy and prediction and recall for each label
    """
    log("----- Confusion Matrix -----")
    matrix = confusion_matrix( test_subset )
    log("%s", ( pandas.DataFrame( matrix ) ) )
    log("Accuracy : %0.2f", ( accuracy(matrix) ) )

    for i in range(NUM_LABEL):
        log("Label %d : precision: %.2f \t recall: %.2f",
            ( i, precision( matrix, i ), recall( matrix, i ) )
        )

    log("----------------")
    log("Time spent : %.0f sec", ( end_compare - start_compare ) )

def cosine_dist( v1, v2 ):
    """ Compute Cosine distance between 2 vectors

    Arguments:
    v1,v2 -- 2 vectors

    Return: consine distance
    """

    dot_product = 0
    for i in range(len(v1.features)):
        dot_product += ( v1.features[i]*v2.features[i] )

    similarity = dot_product / ( 1.0*( v1.length * v2.length ) )

    return 1 - similarity


def confusion_matrix(data):
    """ Create confusion matrix table

    Arguments:
    data -- data with predicted label and actual label

    Returns:
    """

    """ Initialize empty confusion matrix """
    matrix = []
    for r in range(NUM_LABEL):
        matrix.append([0]*NUM_LABEL)

    for i in data:
        temp = matrix[i.label][i.predicted_label]
        matrix[i.label][i.predicted_label] = temp + 1

    return matrix

def precision( confusion_matrix, class_name ):
    bucket = []
    for i in range(NUM_LABEL):
        bucket.append( confusion_matrix[i][class_name] )

    all_predict = sum(bucket)
    correct_predict = confusion_matrix[class_name][class_name]

    precision = 0
    if( all_predict != 0 ):
        precision = correct_predict*1.0 / all_predict

    return precision;

def recall( confusion_matrix, class_name ):
    bucket = confusion_matrix[class_name]

    instance_in_class = sum(bucket)
    correct_predict = confusion_matrix[class_name][class_name]

    recall = 0
    if( instance_in_class != 0 ) :
        recall = 1.0*correct_predict/instance_in_class

    return recall

def accuracy( confusion_matrix ):
    total_instance = 0
    correct_predict = 0
    for i in range(NUM_LABEL):
        total_instance  = total_instance + sum( confusion_matrix[i] )
        correct_predict = correct_predict + confusion_matrix[i][i]
    return correct_predict*1.0 / total_instance

def log( format, data=() ):
    text = format % data
    print text


if __name__ == '__main__':
    run()
