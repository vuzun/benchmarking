from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import time
import pickle
from sklearn.metrics import roc_auc_score

with open("bigdata_for_svm.pickle","rb") as f:
    bigdata_training=pickle.load(f).values
    bigdata_testing=pickle.load(f).values
    class_training=pickle.load(f)
    class_testing=pickle.load(f)

def a_run(feature_number):
    print "Trying for:",feature_number
    the_train=bigdata_training[:,:feature_number]
    the_test=bigdata_testing[:,:feature_number]
    
    the_prediction=None
    auc_score=None
    
    
    starting=time.time()
    
    try:
        an_svm=RandomForestClassifier(n_estimators=100)
        an_svm.fit(the_train, class_training)
        
        the_prediction=an_svm.predict_proba(the_test)
        auc_score=roc_auc_score(class_testing, the_prediction[:,1]) #1 to select class 1 probabilities
    except Exception as e:
        print "This caused a failure:"+e
        
    elapsed=time.time() - starting
    
    return (elapsed, feature_number, auc_score, the_prediction[:,1])
    
    
all_f=bigdata_training.shape[1]

feature_numbers=[int(all_f*i) for i in (0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.9,1)]
res_list=map(a_run,feature_numbers)

with open("../data/reslist_skl_rf_feature_iter.pickle","wb") as r:
    pickle.dump(res_list, r)
