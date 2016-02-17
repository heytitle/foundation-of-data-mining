import math

class Instance:
    def __init__( self, features, label_array=[0]*9, label=None, predicted_label=None ):
        self.features = features


        sos = 0
        for d in self.features:
            sos += math.pow(d,2)
        self.length = math.sqrt(sos)

        if( sum(label_array) > 0  ):
            for j in range( len(label_array) ):
                if( label_array[j] == 1):
                    self.label = j
                    break
        else:
            self.label = label

        if( predicted_label != None ):
            self.predicted_label = predicted_label
