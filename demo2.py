# Demo 2: Q&A (Q&A like ***base DecisionTree***, not LearningDecisionTree)
# ========================================================================
# python                                  # Start python
execfile('initLDT_stage3cancer.py')     # Create LDT with stage3cancer data
dt, root_node = dt1()

ans = interview()
ans
ans['solution_path']
ans['"pgstat"=0']
