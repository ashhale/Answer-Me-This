#!/usr/bin/env python
#
# animals.py
# By Ashley Hale
#
# Experiment with the animals.csv training set
import LearningDecisionTree
from LearningDecisionTree import *

if __name__ == '__main__':
    '''
    This test code assumes animals.csv exists in the current directory.
    '''
    dt = LearningDecisionTree( training_datafile = "animals.csv",
                                csv_class_column_index = 2,
                                csv_columns_for_features = [3,4,5,6,7,8],
                                entropy_threshold = 0.01,
                                max_depth_desired = 8,
                                # csv_cleanup_needed = 1,
                                # save_training_datafile = "new_animals.csv",
                                # debug1 = 'debug1',
                              )
    
    # printBox is a helper function in the code above and is not needed
    # for anything in the LearningDecisionTree - it's just useful for printing
    printBox("Testing with training data file '%s'" % dt._training_datafile)
    
    # ALL of these calls need to be done to set up the decision tree, except
    # show_training_data.
    #
    # root_node will be of type DTNode:
    # https://engineering.purdue.edu/kak/distDT/DecisionTree-3.4.3.html#DTNode
    dt.get_training_data()                                  # Required
    dt.calculate_first_order_probabilities()                # Required
    dt.calculate_class_priors()                             # Required
    # dt.show_training_data()                                 # Optional        
    root_node = dt.construct_decision_tree_classifier()     # Required

