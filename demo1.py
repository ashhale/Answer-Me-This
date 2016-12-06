# Demo 1: Tree Diagram
# ====================

# Start with 4 things open: 2 command prompts and a windows explorer all
# in the same directory, plus a browser open
# 
# In Command Prompt 1:
# python                                  # Start python
execfile('initLDT_stage3cancer.py')     # Create LDT with stage3cancer data
dt, root = dt1()                            # dt1() reads from stage3cancer.csv
                                            # and writes to stage3cancer_v2.csv
dt.getDOT(outFile = 'test.dot')         # Generate file test.dot


# In Command Prompt 2:
# dot -Tsvg -O test.dot
# 
# 
# In Windows Explorer Window:
# Double-click test.dot.svg               # Or drag-drop to browser
# 
# 
# In Browser:
# View, discuss, zoom tree diagram
