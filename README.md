# cfe
Chan Zuckerberg Initiative: Citation Field Extraction

## Directory structure

```
cfe
|
+-- data
|  
+-- src
```

## Dependencies
- Python 2.7
- numpy
- scipy
- pickle
- gensim
- validators
- simstring
- tensorflow
- tensorflow-gpu
- sklearn
- matplotlib
- seaborn
- pandas

## Steps to run the code:

### LSTM
1. Install all the dependencies, if necessary.
2. The LSTM_Model class in lstm_inference.py takes 3 parameters: train_set, use_gpu and do_test.
3. train_set can take 4 values:
```
0:'umass'
1:'combined'
2:'heldout'
3:'umass_heldout'
```
0 -- training on umass, testing on umass, 1 -- training on combined, testing on combined, 2 -- training on combined, testing on heldout and 3 -- training on umass, testing on heldout.
4. use_gpu is a boolean parameter, and it refers to whether gpu is to be used or cpu.
5. do_test is also a boolean parameter. A True value for do_test means that the training model will predict labels for the test set. A False value means that first the model will be trained, and then tested on the test set.
```
model=LSTM_Model(1,use_gpu=False,do_test=True)
model.run_lstm()
```
Here, model=LSTM_Model(1,use_gpu=False,do_test=True) will use the model trained on the combined test set(1).
6. test_on_testset method of the LSTM_Model class takes 3 parameters. The first two parameters refer to the X and y of the test set to be used for evaluation. The third parameter can take 4 values: 0 through 3. This values refers to the model that is to be used for evaluation. 0 and 3 refer to the model trained on umass training set dataset, and 1 and 2 refer to the model trained on the combined training set. Replace self.X_final_test and self.y_final_test with self.X_test and self.y_test respectively to test on the umass or combined test sets respectively.
```
def run_lstm(self):
    self.get_data()
    self.make_model()

    if self.do_test==False:
      self.train()
      self.test_on_testset(self.X_test,self.y_test,self.train_set)
    else:
      self.test_on_testset(self.X_final_test,self.y_final_test,2) # Replace self.X_final_test and self.y_final_test with self.X_test and self.y_test respectively to test on the umass or combined test sets respectively.
```
7. Run lstm_inference.py. This will give the token-level scores on the test set. This will also generate confusion matrix for the particular experiment. All the confusion matrices are stored as .png images in the same directory (src/). For each experiment, a normalized and an unnormalized confusion matrices are generated. They are named based on the experiment. For eg, lstm_cm_norm_heldout.png refers to the "train on combined, test on heldout" experiment.
8. To get segment-level F1 scores, run segment_level_report.py.
```
### Train on umass, test on umass
#data_zip=np.load('../../data/we_npy_no_bio/umass_dataset.npz')
#y_test=data_zip['y_test']
#y_pred = np.load('../../data/lstm_test_result_umass.npy')
```
```
### Train on umass, test on heldout
#data_zip=np.load('../../data/we_npy_no_bio/final_test.npz')
#y_test=data_zip['y_test']
#y_pred=np.load('../../data/lstm_test_result_umass_heldout.npy')
```
```
### Train on combined, test on combined
data_zip=np.load('../../data/we_npy_no_bio/combined_dataset.npz')
y_test=data_zip['combined_y_test']
y_pred=np.load('../../data/lstm_test_result_combined.npy')
```
```
### Train on combined, test on heldout
#data_zip=np.load('../../data/we_npy_no_bio/final_test.npz')
#y_test=data_zip['y_test']
#y_pred=np.load('../../data/lstm_test_result_heldout.npy')
```
Each of the above-mentioned segment of code in segment_level_report.py refers to one of the 4 experiments. Uncomment the necessary segment depending on the type of experiment. As shown above, the third segment is uncommented, and this will generate the segment-level scores on the combined test set, using the model trained on the combined training set.
9. All the scores are stored in data/results.json and all the optimal hyperparameter values are stored in data/params.json.  
