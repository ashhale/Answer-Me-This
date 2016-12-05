#!/usr/bin/env python
#
# LearningDecisionTree.py
# By Ashley Hale
#
# Implement a "Learning" Decision Tree based on the DecisionTree from
#    https://pypi.python.org/pypi/DecisionTree/3.4.3
#    https://engineering.purdue.edu/kak/distDT/DecisionTree-3.4.3.html
#
# Requirements:
#   * Python 2.7 (untested with Python 3)
#   * DecisionTree (recommend installing via: pip install DecisionTree)
#
# Recommended:
#   * Graphviz from http://www.graphviz.org/Download.php
#     Graphviz is not needed directly by this file, but the function
#     LearningDecisionTree.getDOT() returns a DOT language description of the
#     tree which Graphviz (specifically the "dot" command) can use to
#     generate a very nice graphical tree. For example, assuming getDOT()
#     wrote its output to the file my_graph.dot, you can generate the file
#     my_graph.dot.svg via the command line: dot -Tsvg -O my_graph.dot

import DecisionTree
from DecisionTree.DecisionTree import sample_index
from DecisionTree.DecisionTree import convert
from __builtin__ import True, str
import sys
import re

class LearningDecisionTree(DecisionTree.DecisionTree):
    '''
    An extension of DecisionTree with additional support to make it easier
    for the tree to "learn."
    
    In addition to the arguments for a DecisionTree, LearningDecisionTree's
    constructor also accepts:
    
    save_training_datafile: string naming a file for where to write out
        updated training data; if not supplied, sys.stdout will be used
    '''
    def __init__(self, *args, **kwargs  ):
        # super() throws exceptions for unknown arguments, so we must
        # extract new LearningDecisionTree arguments before calling super()
        self._save_training_datafile = None
        if 'save_training_datafile' in kwargs:
            self._save_training_datafile = kwargs.pop('save_training_datafile')
        super(LearningDecisionTree, self).__init__(*args, **kwargs )

    def getDOT(self, prefix = 'node [shape=box]\n',
               outFile = None, allProbsOnLeaf = False):
        '''
        Return a string with the DOT language for the tree, with optional
        output to a file.
        
        See http://www.graphviz.org/pdf/dotguide.pdf
        
        This web page considers how to efficiently build a large string one
        piece at a time, and this code uses, "Method 4: Build a list of strings,
        then join it": https://waymoot.org/home/python_string/
        '''
        dotList = ['digraph G {\n  ', prefix]
        dotList.append(self.recurseDOT(self._root_node, allProbsOnLeaf))
        dotList.append('}\n')
        dot = ''.join(dotList)
        
        if outFile:
            FILE = open(outFile, 'w')
            FILE.write(dot)
            FILE.close()

        return dot
        
    def recurseDOT(self, node, allProbsOnLeaf = False):
        '''
        Recursive helper function used by getDOT.

        Produce a DOT line for a label for node. Non-leaf nodes display the
        feature they test with. By default, leaf nodes display the most likely
        class name and its probability. If allProbsOnLeaf is True, leaf nodes
        display the probabilities of all classes, with no class names. All nodes
        display the list of features and values/ranges that lead to them.
        
        If node has children, produce a DOT line for each edge from node to
        each child, then recursively call this function for each child.
        '''
        leaf = len(node.get_children()) == 0
        featureOrProbs = ''
        if leaf:
            classProbs = node.get_class_probabilities()
            probsDisplay = ["%0.3f" %
                                               x for x in classProbs]
            i = classProbs.index(max(classProbs))

            if allProbsOnLeaf:
                # Display the probabilities of all classes, with no class names
                featureOrProbs = str(probsDisplay)
            else:
                # Display the most likely class name and its probability
                featureOrProbs = '[%s => %s]' % (node.get_class_names()[i].
                                                 replace('"', '\\"'),
                                                 probsDisplay[i])
        else:
            featureOrProbs = node.get_feature().replace('"', '\\"')

        dotList = []
        # Add label info, even for leaves
        dotList.append(
            '  "NODE %u" [label="%u: %s\\n%s\\l"%s]\n'
            % (node.get_serial_num(),
               node.get_serial_num(),
               featureOrProbs,
               str(node.get_branch_features_and_values_or_thresholds()).replace(
                   '"', '\\"').replace(', ', ',\\l'),
               '; style="rounded, bold"' if leaf else ""))

        # Add edges to children
        if not leaf:
            for child in node.get_children():
                dotList.append('    "NODE %u" -> "NODE %u";\n'
                           % (node.get_serial_num(), child.get_serial_num()))
        
        # Recursively add labels, and possibly edges, for each child
        for child in node.get_children():
            dotList.append(self.recurseDOT(child, allProbsOnLeaf))
        
        return ''.join(dotList)

    def getQuestionAndValues(self, node):
        '''
        Return info about a DTNode's feature test, sufficient to create a
        nicely formatted question or prompt and error-check the user's input.
        
        A 3-tuple is returned as follows:
        [0] is True if the feature is symbolic, False if the feature is Numeric
        [1] an array of either (a) the possible values for a symbolic feature
        or (b) the [low, high] range for a numeric feature
        [2] is a string with a question about a feature to select a child node
        
        If node is a leaf, return None.

        For a discussion about symbolic vs. numeric features, see
        https://engineering.purdue.edu/kak/distDT/DecisionTree-3.4.3.html#5
        
        TODO: Doc Note: This changed from checkpoint 2. Need to return a 3-tuple
        to return the type of feature, symbolic or numeric. In checkpoint 2,
        returned a 2-tuple.
        '''

        # To determine what fields to access, looked at
        # DecisionTree.classify_by_asking_questions and
        # DecisionTree.interactive_recursive_descent_for_classification
        if len(node.get_children()) == 0:
            return None

        feature = node.get_feature()
        symbolic = self.isSymbolicFeature(feature)
        question = "What is the value for the feature '%s'?" % feature

        values = None
        if symbolic:
            values = self._features_and_unique_values_dict[feature]
        else:
            values = self._numeric_features_valuerange_dict[feature]

        return (symbolic, values, question)

    def chooseChild(self, node, value):
        '''
        Given a DTNode and a value for its feature test, return the selected
        child DTNode.
        
        If node is a leaf or no child matched value, return None.
        The value must already exist in the training data.

        TODO: Doc Note: This changed from checkpoint 2. Feature was an
        unnecessary argument so removed it. In checkpoint 2, the function
        declaration was: def chooseChild(self, node, feature, value):
        '''
        if len(node.get_children()) == 0:
            return None
        
        test_value = convert(value)
        feature = node.get_feature()
        symbolic = self.isSymbolicFeature(feature)
        values = None
        if symbolic:
            values = self._features_and_unique_values_dict[feature]
        else:
            values = self._numeric_features_valuerange_dict[feature]

        if symbolic and test_value not in values:
            raise Exception(
                "Value '%s' not in known values %s for feature '%s' at node %u"
                % (value, values, feature, node.get_serial_num()))
        elif not symbolic and ( test_value < values[0] or
                                test_value > values[1]):
            raise Exception(
                "Value '%s' out of range %s for feature '%s' at node %u"
                % (value, values, feature, node.get_serial_num()))

        if symbolic:
            for child in node.get_children():
                for fv in child.get_branch_features_and_values_or_thresholds():
                    f, v = fv.split('=')
                    if f == feature and convert(v) == test_value:
                        return child
        else:
            found = False
            pattern1 = r'(.+)<(.+)'
            pattern2 = r'(.+)>(.+)'

            for child in node.get_children():
                for fv in child.get_branch_features_and_values_or_thresholds():
                    if re.search(pattern1, fv):
                        m = re.search(pattern1, fv)
                        f,threshold = m.group(1),m.group(2)
                        if test_value <= float(threshold):
                            found = True
                        else:
                            found = False
                    elif re.search(pattern2, fv):
                        m = re.search(pattern2, fv)
                        f,threshold = m.group(1),m.group(2)
                        if test_value > float(threshold):
                            found = True
                        else:
                            found = False
                if found:
                    return child
        
        return None

    def getDeepestChildForFeature(self, node, value):
        '''
        Given a DTNode and a value for its feature test, return the deepest
        descendant DTNode whose parent tests on the same feature and matches
        the given value.
        
        If a leaf is reached that tests on the same feature, that leaf is
        returned. If node itself is a leaf or no child matched value, return
        None.
        
        The value must already exist in the training data.
        '''
        feature = node.get_feature()

        deepest = child = self.chooseChild(node, value)
        while child and child.get_feature() == feature:
            deepest = child
            child = self.chooseChild(child, value)
        
        if child: deepest = child
        return deepest

    def addTrainingRecord(self, newRecord):
        '''
        Accept a new training record as a python Dictionary.

        Keys must be the same as the existing keys in the training data
        (i.e., the same names for the class and features).
        
        Return true or false on success or failure.
        '''
        # TODO: Stub function, implement for real
        return True

    def getFeatureValues(self, node):
        '''
        Given a DTNode, return a Dictionary that includes the feature values
        supplied so far to reach that node.
        
        The Dictionary will include the class and all features, with some values
        missing as indicated by the value None.
        '''
        # TODO: Stub function, implement for real
        return {"class":"human", "height":"short", "size":None}

    def saveTrainingData(self):
        '''
        Save the training data to the save_training_datafile or sys.stdout.
        
        If save_training_datafile was not specified in the constructor, then
        write the training data to sys.stdout.

        TODO: Currently data may be written in different column order
        than the original training file.

        Return true or false on success or failure.
        '''
        
        FILE = sys.stdout
        if self._save_training_datafile:
            FILE = open(self._save_training_datafile, 'w')
        
        # Write the header row
        # TODO: Do we need to handle multiple class columns?
        FILE.write(',%s' % csvFormat(self._class_names[0].split('=')[0]))
        features = self._features_and_values_dict.keys()
        for feature in sorted(features):
            FILE.write(',%s' % csvFormat(feature))
        FILE.write('\n')
        
        # Write the training records. Each record is has index in first column,
        # class value in second column, then feature values in remaining
        # columns.
        #
        # TODO: Maybe preserve original column order - see self._feature_names,
        # self._csv_class_column_index, self._csv_columns_for_features
        for item in sorted(self._training_data_dict.items(),
                           key = lambda x: sample_index(x[0]) ):
            FILE.write('%s' % csvFormat(sample_index(item[0])))
            FILE.write(',%s' % csvFormat(self._samples_class_label_dict.
                                          get(item[0]).split('=')[1]))
            for feature in sorted(item[1]):
                FILE.write(',%s' % csvFormat(feature.split('=')[1]))
            FILE.write('\n')

        if FILE is not sys.stdout:
            FILE.close()
        
        return True

#----------------  LearningDecisionTree Class Helper Functions  ----------------

    def isNumericFeature(self, feature):
        '''
        Given a feature name, return True if the LearningDecisionTree is
        treating it as a numeric feature, or False if treating it as a symbolic
        feature.

        See https://engineering.purdue.edu/kak/distDT/DecisionTree-3.4.3.html#5
        '''
        if feature in self._prob_distribution_numeric_features_dict:
            return True
        return False

    def isSymbolicFeature(self, feature):
        '''
        Given a feature name, return True if the LearningDecisionTree is
        treating it as a symbolic feature, or False if treating it as a numeric
        feature.

        See https://engineering.purdue.edu/kak/distDT/DecisionTree-3.4.3.html#5
        '''
        return not self.isNumericFeature(feature)
    
#--------------------------  Global Helper Functions  --------------------------

def csvFormat(value):
    '''
    Given a value, interpret it as a string and then minimally wrap and escape
    quotes as needed for writing the value as a single item in a CSV record.

    Double up any double quotes in value, then if value contains any quotes or
    commas, wrap the entire value in a pair of double quotes.
    
    As a special case, if value is already surrounded in double quotes, and
    it's properly formatted as a CSV value, then return the original value
    (to avoid layering on an extra set of quotes and escaping of quotes).
    
    Return value as a properly formatted CSV item, as a string.
    '''
    item = str(value)

    # If surrounding quotes already exist, keep them but avoid adding more
    # if value is already properly csv formatted
    if len(item) > 1 and item.startswith('"') and item.endswith('"'):
        item = item[1:-1]               # Strip the surrounding quotes
        item = item.replace('""', '')   #   and any embedded pairs of quotes
        if '"' not in item:             # If no quotes remain
            return str(value)           #   then value is already formatted
        else:                           # otherwise
            item = str(value)           #   value is not properly formatted
        
    item = item.replace('"', '""')    # Escape double quotes by doubling them
    if '"' in item or ',' in item:
        return '"' + item + '"'
    return item    

def printBox(message = '', width = 79, boxChar = '#', outFile = None):
    '''
    Print a text message with a "box" around it.
    
    If message contains '\n' characters, the message is split on multiple lines
    accordingly. Each line should be shorter than width, but this is not
    enforced.

    If outFile is provided, output is written to that file, otherwise output
    is written to sys.stdout.
    There is no fancy error checking for '\r' or '\t' or other odd characters
    in message.
    '''
    FILE = sys.stdout
    if outFile:
        FILE = open(outFile, 'w')
    
    FILE.write(boxChar * width + '\n')
    for line in message.split('\n'):
        FILE.write(boxChar + ' ' + line)
        FILE.write(' ' * (width - len(line) - 3) + boxChar + '\n')
    FILE.write(boxChar * width + '\n')
    
    if FILE is not sys.stdout:
        FILE.close()

#---------------------------------  Test Code  ---------------------------------

if __name__ == '__main__':
    '''
    This test code assumes stage3cancer.csv exists in the current directory.
    This file is supplied with DecisionTree in the Examples directory.
    https://engineering.purdue.edu/kak/distDT/DecisionTree-3.4.3.html#15
    
    To understand test results here, run classify_by_asking_questions.py and
    answer the questions with the values 2.0, 4.0, 2.6, 47. Compare the results
    there and from running these tests. It's also useful to look at the tree
    displayed by classify_by_asking_questions.py and compare it to a tree
    generated from the output of LearningDecisionTree.getDOT().
    
    '''
    #===========================================================================
    # dt = LearningDecisionTree( training_datafile = "training_symbolic.csv",
    #                             csv_class_column_index = 1,
    #                             csv_columns_for_features = [2,3,4,5],
    #                             entropy_threshold = 0.01,
    #                             max_depth_desired = 5,
    #                             csv_cleanup_needed = 1,
    #                             save_training_datafile = "new_training_symbolic.csv",
    #                             # debug1 = 'debug1',
    #                           )
    #===========================================================================

    ############################################################################
    #***************************************************************************
    #* This first section needs to be done to set up any decision tree.
    #*
    #* TODO: Until addTrainingRecord() is completed, the save_training_datafile
    #*  argument can be ignored.
    #***************************************************************************
    ############################################################################
    
    # Create the uninitialized decision tree. save_training_datafile
    # specifies the name of a file to save updated training data. All other
    # constructor arguments are passed to the underlying DecisionTree class.
    # See https://engineering.purdue.edu/kak/distDT/DecisionTree-3.4.3.html#2
    # dt will be of type LearningDecisionTree, defined in this file. It is a
    # subclass of DecisionTree, which is described here:
    # https://engineering.purdue.edu/kak/distDT/DecisionTree-3.4.3.html#DecisionTree
    dt = LearningDecisionTree( training_datafile = "stage3cancer.csv",
                                csv_class_column_index = 2,
                                csv_columns_for_features = [3,4,5,6,7,8],
                                entropy_threshold = 0.01,
                                max_depth_desired = 8,
                                # csv_cleanup_needed = 1,
                                save_training_datafile = "new_stage3cancer.csv",
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
    
    ############################################################################
    # The rest of the code below is used for basic testing of the
    # LearningDecisionTree, and also serves as examples for how to use it.
    ############################################################################

    ############################################################################
    printBox("Test getDOT")
    print dt.getDOT(outFile = 'test_get_dot.dot')
    # print dt.getDOT(outFile = 'test_get_dot.dot', allProbsOnLeaf = True)
    
    ############################################################################
    printBox("Test getQuestionAndValues")
    print "*** Test getQuestionAndValues root_node\n", dt.getQuestionAndValues(
        root_node)
    print "    ", dt.getQuestionAndValues(root_node)[0]
    print "    ", dt.getQuestionAndValues(root_node)[1]
    print "    ", dt.getQuestionAndValues(root_node)[2]
    print "*** Test getQuestionAndValues root_node child[1]\n", \
        dt.getQuestionAndValues(root_node.get_children()[1])
    print "    ", dt.getQuestionAndValues(root_node.get_children()[1])[0]
    print "    ", dt.getQuestionAndValues(root_node.get_children()[1])[1]
    print "    ", dt.getQuestionAndValues(root_node.get_children()[1])[2]
    print "*** Test getQuestionAndValues root_node child[1] child[2]\n", \
        dt.getQuestionAndValues(root_node.get_children()[1].get_children()[2])
    print "    ", dt.getQuestionAndValues(root_node.get_children()[1].get_children()[2])[0]
    print "    ", dt.getQuestionAndValues(root_node.get_children()[1].get_children()[2])[1]
    print "    ", dt.getQuestionAndValues(root_node.get_children()[1].get_children()[2])[2]

    ############################################################################
    printBox("Test chooseChild")
    n = dt.chooseChild(root_node, 2.0)
    n.display_node()
    n = dt.chooseChild(n, 4.0)
    n.display_node()
    n = dt.chooseChild(n, 2.6)
    n.display_node()
    n = dt.chooseChild(n, 2.6)
    n.display_node()
    n = dt.chooseChild(n, 2.6)
    n.display_node()
    n = dt.chooseChild(n, 47)
    n.display_node()
    n = dt.chooseChild(n, 1)
    if n:
        n.display_node()
    else:
        print "No child found\n"

    ############################################################################
    printBox("Test getDeepestChildForFeature")
    n = dt.getDeepestChildForFeature(root_node, 2.0)
    n.display_node()
    n = dt.getDeepestChildForFeature(n, 4.0)
    n.display_node()
    n = dt.getDeepestChildForFeature(n, 2.6)
    n.display_node()
    n = dt.getDeepestChildForFeature(n, 47)
    n.display_node()
    n = dt.getDeepestChildForFeature(n, 1)
    if n:
        n.display_node()
    else:
        print "No child found\n"
    
    ############################################################################
    printBox("Test addTrainingRecord")
    print dt.addTrainingRecord(None)

    ############################################################################
    printBox("Test getFeatureValues")
    print dt.getFeatureValues(root_node)

    ############################################################################
    printBox("Test saveTrainingData")
    if dt._save_training_datafile:
        print "*** save_training_datafile is '%s'\n" % dt._save_training_datafile
    else:
        print "*** save_training_datafile is not specified\n"
    print dt.saveTrainingData()
