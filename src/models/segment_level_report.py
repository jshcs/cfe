import numpy as np
from config import *
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_fscore_support
#load the ground truth and predicted result
#y_test is the ground truth and y_pred is the predict result, both are in token level, have not reshape and squeeze

### umass data
#data_zip=np.load('../../data/we_npy_no_bio/umass_dataset.npz')
#y_test=data_zip['y_test']
#y_pred = np.load('../../data/lstm_test_result_umass.npy')
#y_pred = np.load('../../data/idcnn_test_result_umass.npy')

###	umass_heldout
#data_zip=np.load('../../data/we_npy_no_bio/final_test.npz')
#y_test=data_zip['y_test']
#y_pred=np.load('../../data/lstm_test_result_umass_heldout.npy')
#y_pred=np.load('../../data/idcnn_test_result_umass_heldout.npy')


###	combined test data
data_zip=np.load('../../data/we_npy_no_bio/combined_dataset.npz')
y_test=data_zip['combined_y_test']
y_pred=np.load('../../data/lstm_test_result_combined.npy')
y_pred=np.load('../../data/idcnn_test_result_combined.npy')

###	heldout data
#data_zip=np.load('../../data/we_npy_no_bio/final_test.npz')
#y_test=data_zip['y_test']
#y_pred=np.load('../../data/lstm_test_result_heldout.npy')
#y_pred=np.load('../../data/idcnn_test_result_heldout.npy')


print y_test.shape
print y_pred.shape


#from one_hot to label
y_test = np.argmax(y_test, axis = 2)
y_pred = np.argmax(y_pred, axis = 2)

print y_test.shape
print y_pred.shape


#transform token to segment, each citation string has 6 segment, start position and length in one segment
#the format of segment is string '(pos,length)' for label
def get_segment(token_label):
    seg_result = np.array([['0']*len(labels)])
    for i in range(token_label.shape[0]):
        segments = np.array([['(000,000)']*len(labels)])
        for j in range(token_label.shape[1]):
            if token_label[i][j]<6:
                if j==0 or token_label[i][j-1]!=token_label[i][j]:
                    [pos,length] = segments[0][token_label[i][j]].split(',')
                    pos = '(%d'%(j)
                    segments[0][token_label[i][j]] = pos+','+length
                if j+1 == token_label.shape[1] or token_label[i][j+1]!=token_label[i][j]:
                    [pos,length] = segments[0][token_label[i][j]].split(',')
                    length = '%d)'%(j-int(pos[1:])+1)
                    #if int(pos[1:])>9 and i==3:
                        #print length
                    segments[0][token_label[i][j]] = pos+','+length
        seg_result = np.concatenate((seg_result, segments), axis=0)
        
    return seg_result[1:]


#transform token level to segment level for ground truth and prediction
y_seg_test = get_segment(y_test)
y_seg_pred = get_segment(y_pred)

print y_seg_test.shape
print y_seg_pred.shape


averaging='weighted'


person_truth = np.reshape(y_seg_test[:,0],(1,-1))
person_pred = np.reshape(y_seg_pred[:,0],(1,-1))
person_truth = np.squeeze(person_truth)
person_pred = np.squeeze(person_pred)
clf_rep=precision_recall_fscore_support(person_truth,person_pred,average=averaging)
out_dict={
            "precision":clf_rep[0].round(2)
            ,"recall":clf_rep[1].round(2)
            ,"f1-score":clf_rep[2].round(2)
            ,"support":clf_rep[3]
        }


print 'author precision:\t', out_dict["precision"],'recall:\t', out_dict["recall"],'F1 score:\t',out_dict["f1-score"],'support:\t',out_dict["support"]



#f1 score for title
title_truth = np.reshape(y_seg_test[:,1],(1,-1))
title_pred = np.reshape(y_seg_pred[:,1],(1,-1))
title_truth = np.squeeze(title_truth)
title_pred = np.squeeze(title_pred)

clf_rep=precision_recall_fscore_support(title_truth,title_pred,average=averaging)
out_dict={
            "precision":clf_rep[0].round(2)
            ,"recall":clf_rep[1].round(2)
            ,"f1-score":clf_rep[2].round(2)
            ,"support":clf_rep[3]
        }

print 'title precision:\t', out_dict["precision"],'recall:\t', out_dict["recall"],'F1 score:\t',out_dict["f1-score"],'support:\t',out_dict["support"]




#f1 score for journal
journal_truth = np.reshape(y_seg_test[:,2],(1,-1))
journal_pred = np.reshape(y_seg_pred[:,2],(1,-1))
journal_truth = np.squeeze(journal_truth)
journal_pred = np.squeeze(journal_pred)

clf_rep=precision_recall_fscore_support(journal_truth,journal_pred,average=averaging)
out_dict={
            "precision":clf_rep[0].round(2)
            ,"recall":clf_rep[1].round(2)
            ,"f1-score":clf_rep[2].round(2)
            ,"support":clf_rep[3]
        }

print 'journal precision:\t', out_dict["precision"],'recall:\t', out_dict["recall"],'F1 score:\t',out_dict["f1-score"],'support:\t',out_dict["support"]


#f1 score for year
year_truth = np.reshape(y_seg_test[:,3],(1,-1))
year_pred = np.reshape(y_seg_pred[:,3],(1,-1))
year_truth = np.squeeze(year_truth)
year_pred = np.squeeze(year_pred)


clf_rep=precision_recall_fscore_support(year_truth,year_pred,average=averaging)
out_dict={
            "precision":clf_rep[0].round(2)
            ,"recall":clf_rep[1].round(2)
            ,"f1-score":clf_rep[2].round(2)
            ,"support":clf_rep[3]
        }

print 'year precision:\t', out_dict["precision"],'recall:\t', out_dict["recall"],'F1 score:\t',out_dict["f1-score"],'support:\t',out_dict["support"]


#f1 score for volume
volume_truth = np.reshape(y_seg_test[:,4],(1,-1))
volume_pred = np.reshape(y_seg_pred[:,4],(1,-1))
volume_truth = np.squeeze(volume_truth)
volume_pred = np.squeeze(volume_pred)

clf_rep=precision_recall_fscore_support(volume_truth,volume_pred,average=averaging)
out_dict={
            "precision":clf_rep[0].round(2)
            ,"recall":clf_rep[1].round(2)
            ,"f1-score":clf_rep[2].round(2)
            ,"support":clf_rep[3]
        }


print 'volume precision:\t', out_dict["precision"],'recall:\t', out_dict["recall"],'F1 score:\t',out_dict["f1-score"],'support:\t',out_dict["support"]


#f1 score for pages
pages_truth = np.reshape(y_seg_test[:,5],(1,-1))
pages_pred = np.reshape(y_seg_pred[:,5],(1,-1))
pages_truth = np.squeeze(pages_truth)
pages_pred = np.squeeze(pages_pred)

clf_rep=precision_recall_fscore_support(pages_truth,pages_pred,average=averaging)
out_dict={
            "precision":clf_rep[0].round(2)
            ,"recall":clf_rep[1].round(2)
            ,"f1-score":clf_rep[2].round(2)
            ,"support":clf_rep[3]
        }

print 'pages precision:\t', out_dict["precision"],'recall:\t', out_dict["recall"],'F1 score:\t',out_dict["f1-score"],'support:\t',out_dict["support"]


#f1 score for all labels
overall_truth = np.reshape(y_seg_test,(1,-1))
overall_pred = np.reshape(y_seg_pred,(1,-1))
overall_truth = np.squeeze(overall_truth)
overall_pred = np.squeeze(overall_pred)


clf_rep=precision_recall_fscore_support(overall_truth,overall_pred,average=averaging)
out_dict={
            "precision":clf_rep[0].round(2)
            ,"recall":clf_rep[1].round(2)
            ,"f1-score":clf_rep[2].round(2)
            ,"support":clf_rep[3]
        }

print 'overall precision:\t', out_dict["precision"],'recall:\t', out_dict["recall"],'F1 score:\t',out_dict["f1-score"],'support:\t',out_dict["support"]

