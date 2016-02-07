import mnist_dataloader as data
import math
import time

from instance import Instance

def run():
    ( train, undef, test ) =  data.load_data_wrapper()
    log( "Trainning set : %8d instances", ( len(train) ) )
    log( "Testing set   : %8d instances", ( len(test) ) )

    txt = "Converting %s set to Instance set"
    log( txt, ("training") )
    train = [ Instance( t[0], label_array=t[1] ) for t in train ]

    log( txt, ("test") )
    test  = [ Instance( t[0], label=t[1] ) for t in test ]

    instance = 1
    start_compare = time.time()
    """ Iterate through testing set """
    for i in test[0:1]:
        log("Instance %6d", ( instance ))

        """ Find the closest pair from training set
        """
        closest_pair = train[0]
        max_dist     = cosine_dist( i, closest_pair )

        for j in train[1:]:
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
    return 0

def precision( confusion_matrix, class_name ):
    return 0

def recall( confusion_matrix, class_name ):
    return 0

def accuracy( confusion_matrix, class_name ):
    return 0

def log( format, data=() ):
    text = format % data
    print text


if __name__ == '__main__':
    run()
