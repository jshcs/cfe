import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools

model = 'combT_U'

#load the ground truth and predicted result
#y_test is the ground truth and y_pred is the predict result, both are in token level, have not reshape and squeeze
data_zip=np.load(model +'.npz')
print data_zip.files
y_test=data_zip['arr_1']
y_pred = data_zip['arr_0']

labels = {'person':0,'title':1,'journal':2,'year':3,'volume':4,'pages':5 }

labels_cm = ['person', 'title', 'journal', 'year' , 'volume' , 'pages', 'others']
def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='LSTM Confusion matrix',
                          fname = 'umass',
                          cmap=plt.cm.Reds):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(fname + '.png')
    print title

y_test_cm = []
for a in y_test :
    y_test_cm.extend(a)
print np.shape(y_test_cm)

y_pred_cm = []
for a in y_pred :
    y_pred_cm.extend(a)
print np.shape(y_pred_cm)

cm  = confusion_matrix(y_test_cm, y_pred_cm , labels = labels_cm)

plot_confusion_matrix(cm,labels_cm[:],title = "CRF Confusion Matrix", fname = model)

#from one_hot to label
#y_test = np.argmax(y_test, axis = 2)
#y_pred = np.argmax(y_pred, axis = 2)

#transform token to segment, each citation string has 6 segment, start position and length in one segment
#the format of segment is string '(pos,length)' for label

def get_segment(token_label):
    print np.shape(token_label)
    seg_result = np.array([['0']*len(labels)])
    for i in range(np.shape(token_label)[0]):
        segments = np.array([['(000,000)']*len(labels)])
        for j in range(len(token_label[i])):
            if token_label[i][j] in labels :
                if j==0 or token_label[i][j-1]!=token_label[i][j]:
                    [pos,length] = segments[0][labels[token_label[i][j]]].split(',')
                    pos = '(%d'%(j)
                    segments[0][labels[token_label[i][j]]] = pos+','+length
                if j+1 <= len(token_label[i]) or token_label[i][j+1]!=token_label[i][j]:
                    [pos,length] = segments[0][labels[token_label[i][j]]].split(',')
                    length = '%d)'%(j-int(pos[1:])+1)
                    #if int(pos[1:])>9 and i==3:
                        #print length
                    segments[0][labels[token_label[i][j]]] = pos+','+length
        seg_result = np.concatenate((seg_result, segments), axis=0)
    return seg_result[1:]


#transform token level to segment level for ground truth and prediction
y_seg_test = get_segment(y_test)
y_seg_pred = get_segment(y_pred)

print y_seg_test.shape
print y_seg_pred.shape


#f1 score for author
person_truth = np.reshape(y_seg_test[:,0],(1,-1))
person_pred = np.reshape(y_seg_pred[:,0],(1,-1))
person_truth = np.squeeze(person_truth)
person_pred = np.squeeze(person_pred)
#print 'person_truth',person_truth.shape,'person_pred',person_pred.shape

f1_person = f1_score(person_truth,person_pred,average='weighted')
prescision_person = precision_score(person_truth, person_pred, average='weighted') 
recall_person = recall_score(person_truth, person_pred, average='weighted')  

print 'author ', prescision_person,' ', recall_person,' ',f1_person


#f1 score for title
title_truth = np.reshape(y_seg_test[:,1],(1,-1))
title_pred = np.reshape(y_seg_pred[:,1],(1,-1))
title_truth = np.squeeze(title_truth)
title_pred = np.squeeze(title_pred)
#print 'title_truth',title_truth.shape,'title_pred',title_pred.shape

f1_title = f1_score(title_truth,title_pred,average='weighted')
prescision_title = precision_score(title_truth, title_pred, average='weighted')
recall_title = recall_score(title_truth, title_pred, average='weighted')

print 'title ', prescision_title ,' ', recall_title ,' ',f1_title


#f1 score for journal
journal_truth = np.reshape(y_seg_test[:,2],(1,-1))
journal_pred = np.reshape(y_seg_pred[:,2],(1,-1))
journal_truth = np.squeeze(journal_truth)
journal_pred = np.squeeze(journal_pred)
#print 'title_truth',journal_truth.shape,'title_pred',journal_pred.shape

f1_journal = f1_score(journal_truth,journal_pred,average='weighted')
prescision_journal = precision_score(journal_truth, journal_pred, average='weighted') 
recall_journal = recall_score(journal_truth, journal_pred, average='weighted')  

print 'journal ', prescision_journal, '  ', recall_journal,' ',f1_journal


#f1 score for year
year_truth = np.reshape(y_seg_test[:,3],(1,-1))
year_pred = np.reshape(y_seg_pred[:,3],(1,-1))
year_truth = np.squeeze(year_truth)
year_pred = np.squeeze(year_pred)
#print 'year_truth',year_truth.shape,'year_pred',year_pred.shape

f1_year = f1_score(year_truth,year_pred,average='weighted')
prescision_year = precision_score(year_truth, year_pred, average='weighted') 
recall_year = recall_score(year_truth, year_pred, average='weighted')  

print 'year ', prescision_year,' ', recall_year,' ',f1_year


#f1 score for volume
volume_truth = np.reshape(y_seg_test[:,4],(1,-1))
volume_pred = np.reshape(y_seg_pred[:,4],(1,-1))
volume_truth = np.squeeze(volume_truth)
volume_pred = np.squeeze(volume_pred)
#print 'volume_truth',volume_truth.shape,'volume_pred',volume_pred.shape

f1_volume = f1_score(volume_truth,volume_pred,average='weighted')
prescision_volume = precision_score(volume_truth, volume_pred, average='weighted') 
recall_volume = recall_score(volume_truth, volume_pred, average='weighted')  

print 'volume ', prescision_volume,' ', recall_volume,' ',f1_volume


#f1 score for pages
pages_truth = np.reshape(y_seg_test[:,5],(1,-1))
pages_pred = np.reshape(y_seg_pred[:,5],(1,-1))
pages_truth = np.squeeze(pages_truth)
pages_pred = np.squeeze(pages_pred)
#print 'pages_truth',pages_truth.shape,'pages_pred',pages_pred.shape

f1_pages = f1_score(pages_truth,pages_pred,average='weighted')
prescision_pages = precision_score(pages_truth, pages_pred, average='weighted') 
recall_pages = recall_score(pages_truth, pages_pred, average='weighted')  

print 'pages ', prescision_pages,' ', recall_pages,' ',f1_pages


#f1 score for all labels
overall_truth = np.reshape(y_seg_test,(1,-1))
overall_pred = np.reshape(y_seg_pred,(1,-1))
overall_truth = np.squeeze(overall_truth)
overall_pred = np.squeeze(overall_pred)
#print 'overall_truth',overall_truth.shape,'overall_pred',overall_pred.shape

f1_overall = f1_score(overall_truth,overall_pred,average='weighted')
prescision_overall = precision_score(overall_truth, overall_pred, average='weighted') 
recall_overall = recall_score(overall_truth, overall_pred, average='weighted')  

print 'overall ', prescision_overall,' ', recall_overall,' ',f1_overall

