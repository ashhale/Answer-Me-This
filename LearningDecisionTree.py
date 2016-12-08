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
import sys
import re
from numbers import Number

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
        # super() throws exceptions for unknown arguments, so we must extract
        # new LearningDecisionTree-specific arguments before calling super()
        self._save_training_datafile = None
        if 'save_training_datafile' in kwargs:
            self._save_training_datafile = kwargs.pop('save_training_datafile')

        super(LearningDecisionTree, self).__init__(*args, **kwargs )

        self._answers = {}              # See startQandA()
        self._path = []                 # See startQandA()
        self._recording = False         # See startQandA()
        self._nodeDict = None           # See getNodeDict()

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
        Recursive helper function used by getDOT().

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
            probsDisplay = ["%0.3f" % x for x in classProbs]
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

    def  getQuestionAndValues(self, node):
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
        Given a DTNode and a value for its feature  test, return the selected
        child DTNode.
        
        If node is a leaf or no child matched value, return None.
        The value must already exist in the training data.
        
        If recording is enabled, will add feature/value to recording
        dictionary. See startQandA().
        
        TODO: Maybe add a new argument (allowNewValues = False) which, if
        True, would skip the tests against known values. If this did not
        cause problems elsewhere, it could be an approach for adding new
        values to a feature in the training data. Would also need to add
        allowNewValues argument to getDeepestChildForFeature.

        TODO: Doc Note: This changed from checkpoint 2. Feature was an
        unnecessary argument so removed it. In checkpoint 2, the function
        declaration was: def chooseChild(self, node, feature, value):
        '''
        if len(node.get_children()) == 0:
            return None
        
        test_value = convert(value)
        feature = node.get_feature()
        if self._recording:
            self._answers[feature] = test_value

        symbolic = self.isSymbolicFeature(feature)
        values = None
        if symbolic:
            values = self._features_and_unique_values_dict[feature]
        else:
            values = self._numeric_features_valuerange_dict[feature]

        # TODO: Maybe skip these tests if new argument allowNewValues == True??
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
                        if self._recording:
                            self._path.append(child.get_serial_num())
                        return child
        else:
            found = False
            for child in node.get_children():
                # Get all the '<' and '>' clauses in the child for feature
                lessThan = [x for x in 
                            child.get_branch_features_and_values_or_thresholds()
                            if x.split('<')[0] == feature]
                greaterThan = [x for x in 
                            child.get_branch_features_and_values_or_thresholds()
                            if x.split('>')[0] == feature]

                if lessThan:
                    # We want to be <= ALL the '<' clauses
                    if test_value <= min(convert(x.split('<')[1])
                                                  for x in lessThan):
                        found = True
                    else:
                        found = False
                        continue        # Failed so don't test greaterThan
                
                if greaterThan:
                    # We want to be > ALL the '>' clauses
                    if test_value > max(convert(x.split('>')[1])
                                                    for x in greaterThan):
                        found = True
                    else:
                        found = False
                        continue        # For symmetry :-)
                
                if found:
                    if self._recording:
                        self._path.append(child.get_serial_num())
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

        TODO: Maybe add a new argument (allowNewValues = False) which, if
        True, would skip the tests against known values. If this did not
        cause problems elsewhere, it could be an approach for adding new
        values to a feature in the training data. Would also need to add
        allowNewValues argument to chooseChild.
        '''
        feature = node.get_feature()

        deepest = child = self.chooseChild(node, value)
        while child and child.get_feature() == feature:
            deepest = child
            child = self.chooseChild(child, value)
        
        if child: deepest = child
        
        if deepest.get_feature() == feature:
            return None
        
        return deepest

    def addTrainingRecord(self, newRecord):
        '''
        Accept a new training record as a python Dictionary, optionally
        regenerating the tree.
        
        If newRecord includes a new feature, existing training records will be
        extended to include the new feature with a value of 'NA'.
        newRecord may include a new value for class or any existing feature. 
        
        TODO: A new feature fails at construct_decision_tree_classifier(),
        need to debug
        TODO: A new class value appears to require that there be no NA fields
        in the new record, but works if no NAs
        New numeric feature value out of previous range seems to work.
        New symbolic feature value seems to work.
        
        After this call, the tree is in an inconsistent state and this should be
        followed by the usual setup function calls, something like:
            dt.calculate_first_order_probabilities()                # Required
            dt.calculate_class_priors()                             # Required
            root_node = dt.construct_decision_tree_classifier()     # Required
        It should be OK to call addTrainingRecord() repeatedly before making
        the three calls above, but other tree operations are not to be trusted
        until the three calls above are completed.
        
        This function will raise Exceptions if the newRecord fails validation.
        TODO: Add notes here about what validation is done.
        
        Returns the record ID of the new record, or zero on failure.

        TODO: Doc Note: In checkpoint 2, code comments said no new feature
        names were permitted, but they are now (except for a bug). Also at
        checkpoint 2, code comments said this function returns True or False.
        '''
        ########################################################################
        # Validate and analyze newRecord
        ########################################################################
        newClassValue = None
        newFeatureNames = []
        newFeatureValues = {}
        className = self._class_names[0].split('=')[0]
        classValues = [self._class_names[x].split('=')[1]
                       for x in range(len(self._class_names))]
        newFeatureDict = dict([(x, newRecord[x]) for x in newRecord.keys() if x != className])
        
        # Ensure newRecord has entries for the class and all existing features
        if className not in newRecord:
            raise Exception('newRecord missing entry for class "%s"'
                            % className)
        for f in self._feature_names:
            if f not in newRecord:
                raise Exception('newRecord missing entry for feature "%s"' % f)
        
        # Determine if there are new class/feature values or new feature names
        if newRecord[className] not in classValues:
            newClassValue = newRecord[className]
        for key in newFeatureDict.keys():
            if key not in self._feature_names:
                newFeatureNames.append(key)
            elif newRecord[key] not in self._features_and_unique_values_dict[key]:
                newFeatureValues.update([(key, newRecord[key])])

        ########################################################################
        # Prepare some values for use below
        ########################################################################
        
        # Existing record IDs are not necessarily sequential, as DecisionTree
        # only requires that they be unique
        newID = max([int(self._samples_class_label_dict.keys()[y].split('_')[1])
                     for y in range(len(self._samples_class_label_dict))]) + 1
        newSampleName = 'sample_' + str(newID)
        newClassLabel = className + '=' + str(newRecord[className])
        
        ########################################################################
        # Need to update 9 variables, as per examination of
        # DecisionTree.get_training_data(). See numbered comments below.
        ########################################################################
        
        # 1. self._how_many_total_training_samples
        #    Simple increment
        self._how_many_total_training_samples += 1
        
        # 2. self._class_names
        #    Add an entry if newRecord contains a new class value
        if newClassValue:
            self._class_names.append(className + '=' + str(newClassValue))
        
        # 3. self._feature_names
        #    Add entries if newRecord contains new feature names
        if newFeatureNames:
            self._feature_names.extend(newFeatureNames)
        
        # 4. self._samples_class_label_dict
        #    Add an entry for newRecord
        self._samples_class_label_dict.update([(newSampleName, newClassLabel)])
        
        # 5. self._training_data_dict
        #    If new feature name, add feature name with default value to all
        #    Add an entry for newRecord
        for f in newFeatureNames: 
            for key in self._training_data_dict:
                self._training_data_dict[key].append(f + '=NA')
            
        self._training_data_dict.update([(newSampleName,
                                          [x + '=' + str(newRecord[x])
                                           for x in newFeatureDict.keys()])])
        
        # 6. self._features_and_values_dict
        #    Add newRecord's value to each entry
        #    If new feature name, add feature name with newRecord's value
        #        and enough copies of default value for all other records
        for f in self._features_and_values_dict:
            self._features_and_values_dict[f].append(newRecord[f])
        
        numNA = self._how_many_total_training_samples - 1
        for f in newFeatureNames:
            self._features_and_values_dict[f] = [newRecord[f]]
            self._features_and_values_dict[f].extend(['NA'
                                                      for x in range(numNA)])
        
        # 7. self._features_and_unique_values_dict
        #    Add newRecord's value to each entry if value not already present
        #    If new feature name, add feature name with newRecord's value
        for f in self._features_and_unique_values_dict:
            if newRecord[f] not in self._features_and_unique_values_dict[f]:
                self._features_and_unique_values_dict[f].append(newRecord[f])

        for f in newFeatureNames:
            self._features_and_unique_values_dict.update([(f, newRecord[f])])
        
        # 8. self._numeric_features_valuerange_dict
        #    For each newRecord value, extend range per feature as needed
        #    If new feature name, add feature name with single-value range
        for f in self._numeric_features_valuerange_dict:
            val = convert(newRecord[f])
            if not isinstance(val, Number):
                # TODO: Do we need something more here or will regenerating
                # the tree harmlessly turn f into a symbolic feature?
                continue        # If newRecord's value is not a number, ditch
            low, high = self._numeric_features_valuerange_dict[f]
            self._numeric_features_valuerange_dict[f] = [min(low, val),
                                                         max(high, val)]

        for f in newFeatureNames:
            self._numeric_features_valuerange_dict[f] = [convert(newRecord[f]),
                                                         convert(newRecord[f])]

        # 9. self._feature_values_how_many_uniques_dict
        #    Update with new len() of each self._features_and_unique_values_dict
        #    If new feature name, add new name with count of 1
        for f in self._feature_values_how_many_uniques_dict:
            self._feature_values_how_many_uniques_dict[f] = \
            len(self._features_and_unique_values_dict[f])
        
        for f in newFeatureNames:
            self._feature_values_how_many_uniques_dict.update([(f, 1)])

        ########################################################################
        # Reset a bunch of "self" fields in DecisionTree, found in the
        # DecisionTree constructor.  Unless we do this, one or more of
        # these calls fail with a variety of errors, like divide by zero:
        #    dt.calculate_first_order_probabilities()                # Required
        #    dt.calculate_class_priors()                             # Required
        #    root_node = dt.construct_decision_tree_classifier()     # Required
        ########################################################################

        self._root_node                                     =      None
        self._probability_cache                             =      {}
        self._entropy_cache                                 =      {}
        self._class_priors_dict                             =      {}
        self._sampling_points_for_numeric_feature_dict      =      {}
        self._prob_distribution_numeric_features_dict       =      {}
        self._histogram_delta_dict                          =      {}
        self._num_of_histogram_bins_dict                    =      {}

        return newID

    def getFeatureValues(self, node):
        '''
        Given a DTNode, return a Dictionary that includes the feature values
        supplied so far to reach that node.
        
        The Dictionary will include the class and all features, with some values
        missing as indicated by the value None.
        
        TODO: Doc Note: This function was included in checkpoint 2 but is no
        longer needed. Instead, use the recording feature and then getQandA()
        and/or seedTrainingRecordWithQandA()
        '''
        # TODO: Stub function, implement for real
        # return {"class":"human", "height":"short", "size":None}
        raise NotImplementedError('Function getFeatureValues is not implemented')

    def saveTrainingData(self):
        '''
        Save the training data to the save_training_datafile or sys.stdout.
        
        If save_training_datafile was not specified in the constructor, then
        write the training data to sys.stdout.

        TODO: Currently data may be written in different column order
        than the original training file.
        
        TODO: Doc Note: This changed from checkpoint 2. If the argument
        save_training_datafile was supplied in the constructor, use that as the
        output file, otherwise write to sys.stdout.
        In checkpoint 2, planned to overwrite existing input training data file. 

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

#------------  LearningDecisionTree Recording Management Functions  ------------

    def startQandA(self):
        '''
        Start recording feature-value pairs and node moves from calls to
        chooseChild(), clearing any previously recorded answers and path.
        '''
        self._answers.clear()
        self._path = [self._root_node.get_serial_num()]
        self._recording = True

    def stopQandA(self):
        '''
        Stop recording feature-value pairs and node moves from calls to
        chooseChild(), keeping any previously recorded answers
        '''
        self._recording = False

    def resumeQandA(self):
        '''
        Resume recording feature-value pairs and node moves from calls to
        chooseChild(), keeping any previously recorded answers
        '''
        self._recording = True

    def getQandA(self):
        '''
        Return a dictionary of the recorded feature-value pairs from calls
        to chooseChild().
        
        The length of this dictionary may be shorter than the length of the
        list from getQandAPath since this dictionary will have only one
        entry per feature tested, regardless of how many nodes were used to
        test that same feature.
        '''
        return self._answers

    def getQandAPath(self):
        '''
        Return a list of the recorded node moves from calls to chooseChild().
        
        If the last item in the list is None, then the responses did not lead
        to an answer, which could mean more training data is needed.
        '''
        return self._path

    def seedTrainingRecordWithQandA(self, classValue = 'NA'):
        '''
        Return a partial training record as a python dictionary, based on the
        recorded Q&A so far and a supplied class value. Missing values are
        seeded as 'NA'.
        
        The classValue is not limited to existing values, so this can be used
        to create new class values if passed to addTrainingRecord().
        '''
        # Start the record with the pair class name:classValue
        newValue = convert(classValue)
        newRecord = {self._class_names[0].split('=')[0] : newValue}
        
        # Add every feature name with the value 'NA'
        for f in self._feature_names:
            newRecord.update([(f, 'NA')])
        
        # Update any feature:value pairs from recording
        newRecord.update(self._answers)
        
        return newRecord
        
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
    
    def getNodeDict(self):
        '''
        Return a dictionary of all Nodes, keyed by the node serial numbers.
        
        This can probably be inefficient so the dictionary is not created until
        this function is called, and is cached for later use.
        '''
        if self._nodeDict:
            return self._nodeDict

        self._nodeDict = {}
        self.recurseNodeDict(self._root_node)
        return self._nodeDict
        
    def recurseNodeDict(self, node):
        '''
        Recursive helper function used by getNodeDict()
        '''
        self._nodeDict[node.get_serial_num()] = node
        if len(node.get_children()) > 0:
            for child in node.get_children():
                self.recurseNodeDict(child)
    
    def train(self, displayMoves = False, displayAnswer = False):
        '''
        Recursively prompt the user to answer questions until reaching a leaf
        node, and return a seeded partial training record (see
        seedTrainingRecordWithQandA).
        
        If displayMoves is True, after each question is answered, a message
        is displayed saying which node was reached.
        
        If displayAnswer is True, before returning the seeded training record,
        print a message describing the answer, including the probability of each
        class value, and the node path followed.
        
        To get the full path of all nodes visited at the end of training,
        including intermediate nodes, see getQandAPath. The last node in
        the path is the answer node, which can be examined via getNodeDict.
        TODO: Consider returning the final DTNode also, perhaps a tuple with
        the final DTNode and the seeded training record.
        
        This is intended as a simple demonstration of how to use the other
        methods in LearningDecisionTree to record Q&A from a user navigating
        the tress to produce the "first draft" of a new training record. A
        more fleshed-out UI would take this seeded training record, prompt the
        user to modify it further, and then add the new training record to the
        training data set (see addTrainingRecord and saveTrainingData).
        '''
        node = self._root_node
        self.startQandA()                       # Start recording
        while len(node.get_children()) > 0:
            if node.get_feature() in self._answers:
                value = self._answers[node.get_feature()]
            else:
                value = self.prompt(node)
            prevNode = node
            node = self.getDeepestChildForFeature(node, value)
            if not node:
                # Answers given do not lead to a match
                self._path.append(None)
                print
                break

            if displayMoves:
                print "%s=%s: Moved from node %d to node %d" % (
                       prevNode.get_feature(),
                       str(value),
                       prevNode.get_serial_num(),
                       node.get_serial_num()),
                # Show intermediate nodes, if any
                prevIdx = self._path.index(prevNode.get_serial_num())
                curIdx = self._path.index(node.get_serial_num())
                interEdges = curIdx - prevIdx - 1
                m = ''
                if interEdges > 0:
                    m += " (via node%s %d" % ('s' if interEdges > 1 else '',
                                               self._path[prevIdx + 1])
                    for i in range(interEdges - 1):
                        m += ", %d" % self._path[prevIdx + i + 2]
                    m += ")"
                print m                         # End of "Moved to node" line
            print                               # Blank between prompts

        if displayAnswer:
            if node:
                classProbs = node.get_class_probabilities()
                probsDisplay = ["%0.3f" % x for x in classProbs]
                print("Classification:\n")
                print("     "  + str.ljust("class name", 30) + "probability")    
                print("     ----------                    -----------")
                for i in range(len(node.get_class_names())):
                    print("     "  + str.ljust(node.get_class_names()[i], 30)
                          + probsDisplay[i])
                print
            else:
                print '*** The responses did not lead to an answer ***'
                print '***  Consider adding a new training record  ***\n'
            
        # Get a partial training record seeded with the Q&A answers and
        # the class with the highest predicted probability (if answer found)
        rec = self.seedTrainingRecordWithQandA()
        if node:
            classProbs = node.get_class_probabilities()
            mostLikely = node.get_class_names()[classProbs.index(max(classProbs))]
            rec[mostLikely.split('=')[0]] = mostLikely.split('=')[1]

        # TODO:Instead of returning just  partial seeded record, make a
        # second loop to prompt the user for new values in the seeded record
        # where the seeded record values are currently NA
        return rec

            
    def prompt(self, node):
        '''
        Prompt the user for a value for the feature test of a single node,
        validating the input and re-prompting for bad input.
        
        Return the user's reply
        
        TODO: Add a nice way for the user to quit instead of answering
        '''
        reply = None
        symPrefix = 'Enter one of'
        numPrefix = "Enter a value in the range"
        symbolic, values, question = self.getQuestionAndValues(node)
        entryLine = '%s: %s ' % (symPrefix if symbolic else numPrefix, values)

        print question
        while not reply:
            reply = convert(raw_input(entryLine).strip())
            if symbolic and reply not in values:
                reply = None
            elif not symbolic and (reply < values[0] or reply > values[1]):
                reply = None
        
        return reply
                
                
    
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
    # print dt.getDOT(outFile = 'test_get_dot.dot')
    print dt.getDOT(outFile = 'test_get_dot.dot', allProbsOnLeaf = True)
    
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
    printBox("Test getDeepestChildForFeature and recording")
    dt.startQandA()
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
    seed = dt.seedTrainingRecordWithQandA()
    print "New training record seeded from Q&A recording\n"
    print seed
    
    ############################################################################
    printBox("Test addTrainingRecord")
    newRec = dict(seed)
#     newRec['"eet"'] = 2.0               # Fill in NA value
#     newRec['"ploidy"'] = '"diploid"'    # Fill in NA value
    newRec['"pgstat"'] = 0      # Existing class value
#     newRec['"g2"'] = 2.4        # Existing numeric value in previous range [2.4, 54.93]
#     newRec['"pgstat"'] = 2      # New class value
#     newRec['"g2"'] = 2.1        # New numeric value out of previous range [2.4, 54.93]
    newRec['"ploidy"'] = 'blah' # New symbolic value, no surrounding double quotes
#     newRec.update([('weight', 190)])    # New feature, no surrounding double quotes
    print "Proposed new record to add, modified from seed record\n"
    print newRec
    
    print "Added new training record ID %d" % dt.addTrainingRecord(newRec)

    dt.calculate_first_order_probabilities()                # Required
    dt.calculate_class_priors()                             # Required
    # dt.show_training_data()                                 # Optional        
    root_node = dt.construct_decision_tree_classifier()     # Required

    # print dt.getDOT(outFile = 'test_get_dot.dot')
    print dt.getDOT(outFile = 'test_get_dot_v2.dot', allProbsOnLeaf = True)
    
    ############################################################################
    # printBox("Test getFeatureValues")
    # print dt.getFeatureValues(root_node)

    ############################################################################
    printBox("Test saveTrainingData")
    if dt._save_training_datafile:
        print "*** save_training_datafile is '%s'\n" % dt._save_training_datafile
    else:
        print "*** save_training_datafile is not specified\n"
    print dt.saveTrainingData()
