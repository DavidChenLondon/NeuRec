try:
    from NeuRec.evaluator.backend.cpp.uni_evaluator import UniEvaluator
    print("Evaluate model with cpp")
except:
    from NeuRec.evaluator.backend.python.uni_evaluator import UniEvaluator
    print("Evaluate model with python")
