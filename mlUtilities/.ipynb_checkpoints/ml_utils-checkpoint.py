import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score


def first_func():
    print('Hola Mundo')
    
def run_NaiveBayes(X, Y, test=False, val=False):
    '''Run Naive Bayes.  
    Args:
        X: dataframe
        Y: array or Series
        test: If true, will return values for the test data results
        val: if True, will return valures for the validation data results
    Returns: Dictionary with tuples as the values.
    '''
    class_labels = ['Buy', 'Flat', 'Sell']
    x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2)
    x_train, x_test, y_train, y_testv = train_test_split(x_train, y_train, test_size=0.3)
    nB = GaussianNB().fit(x_train, y_train)
    training_results = nB.predict(x_train)
    training_cm = confusion_matrix(y_train, training_results, labels=class_labels)
    test_accuracy = accuracy_score(y_test, test_results)
    if not test and not val:
        return (training_cm, training_accuracy)
    if test:
        test_results = nB.predict(x_test)
        test_cm = confusion_matrix(y_test, test_results, labels=class_labels)
        test_accuracy = accuracy_score(y_test, test_results)
        if not val:
            return {'Training_Results': (training_cm, training_accuracy),
                    'Test_Results': (test_cm, test_accuracy)
                   }
    if val:
        val_results = nB.predict(x_val)
        val_cm = confusion_matrix(y_val, val_results, labels=class_labels)
        val_accuracy = accuracy_score(y_val, val_results)
        if not test:
            return {'Training_Results': (training_cm, training_accuracy),
                    'Val_Results': (val_cm, val_accuracy)
                   }
        return {'Training_Results': (training_cm, training_accuracy),
                'Test_Results': (test_cm, test_accuracy),
                'Val_Results': (val_cm, val_accuracy)
               }
    
    
def print_results(results_dict, train=True, test=False, val=False, all_metrics=False):
    for target in results_dict:
        print(target)
        if train:
            train_cm = results_dict[target]['Training_Results'][0]
            train_acc = round(results_dict[target]['Training_Results'][1], 2)
            if test:
                test_cm = results_dict[target]['Test_Results'][0]
                test_acc = round(results_dict[target]['Test_Results'][1], 2)
                if val:
                    val_cm = results_dict[target]['Val_Results'][0]
                    val_acc = round(results_dict[target]['Val_Results'][1], 2)
                    print(target)
                    if not all_metrics:
                        print(f'Training: {train_acc}, Test: {test_acc}, Val: {val_acc}')
                    else:
                        print(f'Training: {train_acc}, Test: {test_acc}')
                        print(f'Training: {train_cm[0]}, Test: {test_cm[0]}, Val: {val_cm[0]}')
                        print(f'Training: {train_cm[1]}, Test: {test_cm[1]}, Val: {val_cm[1]}')
                        print(f'Training: {train_cm[2]}, Test: {test_cm[2]}, Val: {val_cm[2]}')
                else:
                    if not all_metrics:
                        print(f'Training: {train_acc}, Test: {test_acc}')
                    else:
                        print(f'Training: {train_acc}, Test: {test_acc}')
                        print(f'Training: {train_cm[0]}, Test: {test_cm[0]}')
                        print(f'Training: {train_cm[1]}, Test: {test_cm[1]}')
                        print(f'Training: {train_cm[2]}, Test: {test_cm[2]}')
            else:
                if not all_metrics:
                    print(f'Training: {train_acc}')
                else:
                    print(f'Training: {train_acc}')
                    print(f'Training: {train_cm[0]}')
                    print(f'Training: {train_cm[1]}')
                    print(f'Training: {train_cm[2]}')
        else:
            print('Functionality for training false is not yet established')
            

def performance_metrics(conf_matrix):
    '''Generate statistics from confusion matrix'''
    tn = conf_matrix[0,0]
    fn = conf_matrix[1,0]
    tp = conf_matrix[1,1]
    fp = conf_matrix[0, 1]
    results = {}
    results['accuracy'] = round((tp+tn)/(tn+fn+tp+fp), 3)
    results['specificity'] = round(tn/(tn+fp), 3)
    results['precision'] = round(tp/(tp+fp), 3)
    results['sensitivity'] = round(tp/(tp+fn), 3)
    results['recall'] = round(results['sensitivity'], 3)
    results['f1'] = round(2*((results['precision']*results['sensitivity'])/
                             (results['precision']+results['sensitivity'])), 3)
    results['balanced_accuracy'] = round((results['sensitivity'] + results['specificity'])/2, 3)
    return results

                                       

def main():
    first_func()
    
if __name__=='__main__':
    main()