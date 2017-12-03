import numpy as np
import pandas as pd

filePath = 'titanic-dataset.csv'


def featureDetection(dataFrame):
    '''Method returns a list of catagorical and coninuous features'''
    features = dataFrame.dtypes
    catagoricalFeatures = dataFrame.dtypes.loc[dataFrame.dtypes == 'object'].index
    continuousFeatures = dataFrame.dtypes.loc[dataFrame.dtypes != 'object'].index
    return catagoricalFeatures, continuousFeatures


def findUnique(dataFrame, catagoricalFeature):
    '''Mehtod to compute number of unique values for each catagorical field'''
    print(dataFrame[catagoricalFeature].apply(lambda x: len(x.unique())))


def frequencyCount(dataFrame, catagoricalFeatures):
    '''Mehtod prints frequency of each unique value for the catagorical class'''
    for x in catagoricalFeatures:
        print(100*(dataFrame[x].value_counts()/dataFrame[x].shape[0]))

def findMissingValues(dataFrame):
    ''' Method returns feature columns with number of null values for each'''
    print(dataFrame.apply(lambda x: sum(x.isnull())))

def imputeCatagoricalFeature(dataFrame, catagoricalFeatures, imputation_values):
    '''Method imputes catagorical features with given values'''
    i = 0;
    while i < len(catagoricalFeatures):
        dataFrame[catagoricalFeatures[i]].fillna(imputation_values[i], inplace = True)
        i += 1

def writeToCsv(dataFrame, fileName, indexVal):
    '''Method writes the dataframe to a .csv file by the given name'''
    dataFrame.to_csv(fileName, index=indexVal)
    print('DataFrame written to a CSV file', fileName)


def main():
    df = pd.DataFrame.from_csv(filePath)
    fields = df.columns # get the names of columns/fields
    print('Number of Columns:', len(fields))
    cat_var, con_var = featureDetection(df) # select catagorical and continuous fields
    print('Catagorical Fields:', cat_var)
    print('Continuous Fields:', con_var)
    findMissingValues(df) # find missing values for each field, if any
    findUnique(df, ['Embarked', 'Cabin']) # passing catagorical field dataframe
    frequencyCount(df, ['Embarked', 'Cabin']) # compute frequecny for each unique value for each catagorical field
    imputeCatagoricalFeature(df, ['Embarked', 'Age'], ['S', np.mean(df['Age'])])
    findMissingValues(df)
    writeToCsv(df, 'new-titanic-dataset.csv', True)


if __name__=="__main__":
    main()