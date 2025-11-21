import numpy as np

def preprocess_scale(sample):

    sample=np.array(sample)

    if sample.size==0:
        raise ValueError("the size of sample is zero")
    
    mean=sample.mean(axis=0)
    std=sample.std(axis=0)
    std[std==0]=1

    return (sample-mean)/std