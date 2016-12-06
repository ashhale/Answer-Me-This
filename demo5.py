# Demo 5: Interactive Q&A with Recording and Seeding New Training Record
# ======================================================================
# python
execfile('initLDT_stage3cancer.py')
dt, root = dt1()                            # dt1() reads from stage3cancer.csv
                                            # and writes to stage3cancer_v2.csv
dt.getDOT(outFile = 'test.dot', allProbsOnLeaf = True)  # Do dot in other window
subprocess.call(['dot', '-Tsvg', '-O', 'test.dot'])     # Call Graphviz to make diagram
#webbrowser.open('file://%s/test.dot.svg' % os.getcwd()) # Open diagram in browser

rec = dt.train(displayMoves = True)         # Interactive!
                                            # Answer 2, 4, 2.6, 47
                                            # Returns a seeded record
print "Seeded training record based on Q&A"
print rec

print "Recorded Q&A: %s" % dt.getQandA()    # Show recorded answers
print "Recorded node path: %s" % dt.getQandAPath()  # Show record node path

print "\nMaking some mods to the seeded record..."
rec['"eet"'] = 2.0                          # Fill in NA value
rec['"ploidy"'] = '"diploid"'               # Fill in NA value
# rec['"pgstat"'] = 0                         # Existing class value
# rec['"g2"'] = 2.4                           # Existing numeric value in previous range [2.4, 54.93]
rec['"pgstat"'] = 2                         # New class value
# rec['"g2"'] = 2.1                           # New numeric value out of previous range [2.4, 54.93]
# rec['"ploidy"'] = 'blah'                    # New symbolic value, no surrounding double quotes
# rec.update([('weight', 190)])               # New feature, no surrounding double quotes

print "New training record to add, modified from seed record"
print rec

print "Added new training record ID %d" % dt.addTrainingRecord(rec)

print "\nRegenerating tree..."

dt.calculate_first_order_probabilities()                # Required
dt.calculate_class_priors()                             # Required
# dt.show_training_data()                                 # Optional        
root_node = dt.construct_decision_tree_classifier()     # Required

# show_training_data prints a LOT - need a nicer way to show new sample_147
# dt.show_training_data()                     # Note new sample_147
dt.getDOT(outFile = 'test_v2.dot', allProbsOnLeaf = True)  # Tree is different!
subprocess.call(['dot', '-Tsvg', '-O', 'test_v2.dot'])  # Call Graphviz to make diagram
#webbrowser.open('file://%s/test_v2.dot.svg' % os.getcwd())  # Open diagram in browser

    # So far, have shown ability to:
    #   * Interactively walk tree while recording answers and node path
    #   * Use recorded steps to create seed training record
    #   * Tweak seed record
    #   * Add training data and regen tree all dynamically with one tree object

dt.saveTrainingData()                       # Save to stage3cancer_v2.csv

dt, root = dt2()                            # dt2() reads stage3cancer_v2.csv
                                            # and writes stage3cancer_v3.csv
# dt.show_training_data()                     # Note new sample_147
dt.getDOT(outFile = 'test_v3.dot', allProbsOnLeaf = True)  # Same as test_v2.dot
subprocess.call(['dot', '-Tsvg', '-O', 'test_v3.dot']) # Call Graphviz to make diagram
#webbrowser.open('file://%s/test_v3.dot.svg' % os.getcwd())  # Open diagram in browser

    # Now confirmed saving new training data works, and it can be read/used
