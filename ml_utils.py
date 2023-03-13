from pyspark.ml.feature import VectorAssembler, RobustScaler
from pyspark.ml import Pipeline

def set_assembler(*args):
    empty = []
    for l in args:
        empty += l
        
    return VectorAssembler() \
        .setHandleInvalid("skip") \
        .setInputCols(empty) \
        .setOutputCol("unscaled_features")
        
def set_scaler():
    return RobustScaler() \
        .setInputCol("unscaled_features") \
        .setOutputCol("features")
        
def set_pipeline(string_indexer=True, *args):
    if string_indexer:
        string_indexer_object = args[0]
        
        args = args[1:] if len(args) > 1 else args
    
        pipe_list = []
        for p in args:
            pipe_list += p
            
        return Pipeline() \
            .setStages(string_indexer_object + pipe_list)
    else:
        pipe_list = []
        for p in args:
            pipe_list += p
            
        return Pipeline() \
            .setStages(pipe_list)