#!/usr/bin/env python

# import LearningDecisionTree
from LearningDecisionTree import *

training_datafile = "stage3cancer.csv"
training_datafile2 = "stage3cancer_v2.csv"
training_datafile3 = "stage3cancer_v3.csv"

def dt1():
    dt = LearningDecisionTree(
                                    training_datafile = training_datafile,
                                    csv_class_column_index = 2,
                                    csv_columns_for_features = [3,4,5,6,7,8],
                                    entropy_threshold = 0.01,
                                    max_depth_desired = 8,
                                    save_training_datafile = training_datafile2,
                                  )
    dt.get_training_data()
    dt.calculate_first_order_probabilities()
    dt.calculate_class_priors()
    #dt.show_training_data()
    root_node = dt.construct_decision_tree_classifier()
    return dt, root_node

def dt2():
    dt = LearningDecisionTree(
                                    training_datafile = training_datafile2,
                                    csv_class_column_index = 1,
                                    csv_columns_for_features = [2,3,4,5,6,7],
                                    entropy_threshold = 0.01,
                                    max_depth_desired = 8,
                                    save_training_datafile = training_datafile3,
                                  )
    dt.get_training_data()
    dt.calculate_first_order_probabilities()
    dt.calculate_class_priors()
    #dt.show_training_data()
    root_node = dt.construct_decision_tree_classifier()
    return dt, root_node

def interview():
    classification = dt.classify_by_asking_questions(root_node)
    save_classification = dict(classification)
    
    solution_path = classification['solution_path']
    del classification['solution_path']
    which_classes = list( classification.keys() )
    which_classes = sorted(which_classes, key=lambda x: classification[x], reverse=True)
    print("\nClassification:\n")
    print("     "  + str.ljust("class name", 30) + "probability")    
    print("     ----------                    -----------")
    for which_class in which_classes:
        if which_class is not 'solution_path':
            print("     "  + str.ljust(which_class, 30) +  str(classification[which_class]))
    
    print("\nSolution path in the decision tree: " + str(solution_path))
    print("\nNumber of nodes created: " + str(root_node.how_many_nodes()))
    
    return save_classification

def walkAndRecord():
    ############################################################################
    printBox("Test getDeepestChildForFeature and recording")

    dt.startQandA()                         # Start recording
    
    # These steps are instead of an interactive Q&A
    n = dt.getDeepestChildForFeature(dt._root_node, 2.0)
    n.display_node()
    n = dt.getDeepestChildForFeature(n, 4.0)
    n.display_node()
    n = dt.getDeepestChildForFeature(n, 2.6)
    n.display_node()
    n = dt.getDeepestChildForFeature(n, 47)
    n.display_node()

    seed = dt.seedTrainingRecordWithQandA() # Get a partial tree based on recording
    print "New training record seeded from Q&A recording\n"
    print seed
    
    ############################################################################
    printBox("Test addTrainingRecord")
    newRec = dict(seed)

    newRec['"eet"'] = 2.0               # Fill in NA value
    newRec['"ploidy"'] = '"diploid"'    # Fill in NA value
#     newRec['"pgstat"'] = 0      # Existing class value
#     newRec['"g2"'] = 2.4        # Existing numeric value in previous range [2.4, 54.93]
    newRec['"pgstat"'] = 2      # New class value
#     newRec['"g2"'] = 2.1        # New numeric value out of previous range [2.4, 54.93]
#     newRec['"ploidy"'] = 'blah' # New symbolic value, no surrounding double quotes
#     newRec.update([('weight', 190)])    # New feature, no surrounding double quotes

    print "Proposed new record to add, modified from seed record\n"
    print newRec
    
    print "Added new training record ID %d" % dt.addTrainingRecord(newRec)

    dt.calculate_first_order_probabilities()                # Required
    dt.calculate_class_priors()                             # Required
    # dt.show_training_data()                                 # Optional        
    root_node = dt.construct_decision_tree_classifier()     # Required

