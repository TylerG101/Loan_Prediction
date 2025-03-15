###Contains function to encode designated columns in a dataframe, and a function to train and evaluate an sklearn
###model of a desired type
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelBinarizer, OrdinalEncoder, OneHotEncoder, StandardScaler


def encode_feature(feature_list:list,
                   encoder,
                   dataframe,
                   Binary=False,
                   OneHot=False):
    '''
    Used for binary and oridnal encoding as well as scaling of specified columns in a dataset
    '''
    n = 0
    for col in feature_list:
        if Binary == True:
            temp_df = pd.DataFrame(encoder.fit_transform(dataframe[col]))
        else:
            temp_df = pd.DataFrame(encoder.fit_transform(dataframe[[col]]))

        if OneHot == True:
            for i in range(dataframe[[col]].nunique().item()):
                temp_df = temp_df.rename(columns={i:(col+str(i))})
                n += 1
        else:
            temp_df = temp_df.rename(columns={0:col})
        
        dataframe = dataframe.drop(col, axis=1)
        dataframe = pd.concat([dataframe, temp_df], axis=1)

    return dataframe




def train_eval_model(model, X_train, y_train, X_test, y_test, savefig=False):
    '''
    Function used to train and evaluate sklearn models
    '''
    model.fit(X_train, y_train)
    print(f'Training accuracy: {(model.score(X_train, y_train) * 100):.3f}')
    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    cm_img = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    print(f'Test accuracy: {(accuracy_score(y_test, preds) * 100):.3f}')

    if savefig == True:
        cm_img.plot().figure_.savefig( 'Confusion_matrix_'+str(model)+'.png')
    else:
        cm_img.plot()
    return 
