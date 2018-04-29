from config import *
import json

# def write_once():
#     results={
#         'CRF':0,
#         'LSTM':0,
#         'Bi-LSTM':0,
#         'ID-CNN':0
#     }
#     with open(RESULTS,'w') as res:
#         json.dump(results,res)

# write_once()


def update_results(model,f1):
    with open(RESULTS,'r') as res:
        results=json.load(res)

    results[model]=f1

    with open(RESULTS,'w') as res:
        json.dump(results,res)

# update_results('LSTM',0.96)
