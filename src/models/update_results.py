from config import *
import json

def write_once_to_results():
    results={
        'CRF':0,
        'LSTM':0,
        'Bi-LSTM':0,
        'ID-CNN':0
    }
    with open(RESULTS,'w') as res:
        json.dump(results,res)

def write_once_to_params():
    results={
        'CRF':{},
        'LSTM':{'lr':0,'d':0,'epoch':0},
        'Bi-LSTM':{},
        'ID-CNN':{}
    }
    with open(PARAMS,'w') as res:
        json.dump(results,res)

# write_once()

# write_once_to_params()


#   dict={"lr":,"d":,"epoch":}
def update_params(model1,model2,dict):
    with open(PARAMS,'r') as res:
        params=json.load(res)

    params[model1][model2]=dict

    with open(PARAMS,'w') as res:
        json.dump(params,res)

def update_results(model1,model2,f1):
    with open(RESULTS,'r') as res:
        results=json.load(res)

    results[model1][model2]=f1

    with open(RESULTS,'w') as res:
        json.dump(results,res)

# update_results('LSTM',0.96)
