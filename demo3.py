# Demo 3: Adding a new Training Record
# ====================================
# python
execfile('initLDT_stage3cancer.py')
dt, root = dt1()                            # dt1() reads from stage3cancer.csv
                                            # and writes to stage3cancer_v2.csv
dt.getDOT(outFile = 'test.dot', allProbsOnLeaf = True)  # Do dot in other window
walkAndRecord()                             # Non-interactive walk through
                                            # questions while recording, and
                                            # regenerates tree
dt.show_training_data()                     # Note new sample_147
dt.getDOT(outFile = 'test_v2.dot', allProbsOnLeaf = True)  # Tree is different!

    # So far, have shown ability to:
    #   * Walk tree while recording actions
    #   * Use recorded steps to create seed training record
    #   * Tweak seed record
    #   * Add training data and regen tree all dynamically with one tree object

dt.saveTrainingData()                       # Save to stage3cancer_v2.csv

dt, root = dt2()                            # dt2() reads stage3cancer_v2.csv
                                            # and writes stage3cancer_v3.csv
dt.show_training_data()                     # Note new sample_147
dt.getDOT(outFile = 'test_v3.dot', allProbsOnLeaf = True)  # Same as test_v2.dot

    # Now confirmed saving new training data works, and it can be read/used
