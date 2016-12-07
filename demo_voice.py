# Demo 1: Tree Diagram
# ====================

# Start with 4 things open: 2 command prompts and a windows explorer all
# in the same directory, plus a browser open
# 
# In Command Prompt 1:
# python                                  # Start python
execfile('initLDT_voice.py')              # Create LDT with voice_small data
dt, root = dt1()                          # dt1() reads from voice_small.csv
                                          # and writes to stage3cancer_v2.csv
dt.getDOT(outFile = 'test_voice.dot')     # Generate file test_voice.dot


# In Command Prompt 2:
# dot -Tsvg -O test_voice.dot
# 
# 
# In Windows Explorer Window:
# Double-click test_voice.dot.svg               # Or drag-drop to browser
# 
# 
# In Browser:
# View, discuss, zoom tree diagram
