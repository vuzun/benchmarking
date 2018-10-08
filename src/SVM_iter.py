from sklearn import svm
import numpy as np
import time
import pickle
from sklearn.metrics import roc_auc_score

with open("../data/bigdata_for_svm.pickle","rb") as f:
    bigdata_training=pickle.load(f).values
    bigdata_testing=pickle.load(f).values
    class_training=pickle.load(f)
    class_testing=pickle.load(f)


#runs the model on the training data and evaluates on the test set
#params: number of features from the original dataset used
#output: time, number of features, accuracy (AUC), test set predictions
def a_run(feature_number):
    the_train=bigdata_training[:,:feature_number]
    the_test=bigdata_testing[:,:feature_number]
    
    the_prediction=None
    auc_score=None
    
    
    starting=time.time()
    
    try:
        an_svm=svm.SVC(probability=True)
        an_svm.fit(the_train, class_training)
        
        the_prediction=an_svm.predict_proba(the_test)
        auc_score=roc_auc_score(class_testing, the_prediction[:,1]) #1 to select class 1 probabilities
    except Exception as e:
        print "This caused a failure:"+e
        
    elapsed=time.time() - starting
    
    return (elapsed, feature_number, auc_score, the_prediction[:,1])
    


all_f=bigdata_training.shape[1]

feature_numbers=[int(all_f*i) for i in (0.25,0.5,0.75,1)]
reslist_svm_feature_iter=map(a_run,feature_numbers)


with open("../data/reslist_svm_iter.pickle","wb") as f:
    pickle.dump(reslist_svm_feature_iter, f)

