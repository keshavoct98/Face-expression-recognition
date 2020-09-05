import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def preprocess(path_to_csv):
    '''preprocessing on csv dataset'''
    
    df = pd.read_csv(path_to_csv)
    print('\n==============================DATAFRAME================================')
    print(df.head(), '\n')
    
    print('Usage classes - ', df['Usage'].unique())
    print('Emotion classes - ', df['emotion'].unique())
    print('Image dimension - ', len(df['pixels'][0].split())**0.5, '\n')
    
    print('Shape -', df.shape)
    print('Training set - ', (df['Usage'] == 'Training').sum())
    print('Public test set - ', (df['Usage'] == 'PublicTest').sum())
    print('Private test set - ', (df['Usage'] == 'PrivateTest').sum(), '\n')
    
    fig, ax = plt.subplots(2, 2, figsize=(8, 7))
    for i in range(2):
        for j in range(2):
            pixel_list = df['pixels'][2*i+j].split()
            img = np.reshape(pixel_list, (48, 48)).astype('int')
            ax[i, j].set_title('Image-'+str(2*i+j))
            ax[i, j].imshow(img, cmap = 'gray')
    plt.savefig('weights/sample_images.png')
    plt.show()
    
    return df
  
    
def train_val_test_split(path_to_csv):
    '''Splits data into train, test and
    validation split.''' 
    
    df = preprocess(path_to_csv)
    x_train, x_val, x_test = [], [], []
    y_train, y_val, y_test = [], [], []

    for i in range(df.shape[0]):
        pixel_list = df['pixels'][i].split()
        img = np.reshape(pixel_list, (48, 48, 1)).astype('float')
        img = img/255

        if df['Usage'][i] == 'Training':
            x_train.append(img)
            y_train.append(df['emotion'][i])
        elif df['Usage'][i] == 'PublicTest':
            x_val.append(img)
            y_val.append(df['emotion'][i])
        else:
            x_test.append(img)
            y_test.append(df['emotion'][i])

    x_train, x_val, x_test = np.array(x_train), np.array(x_val), np.array(x_test)
    y_train, y_val, y_test = np.array(y_train), np.array(y_val), np.array(y_test)
    y_train, y_val, y_test = pd.get_dummies(y_train), pd.get_dummies(y_val), pd.get_dummies(y_test)
    
    return x_train, y_train, x_val, y_val, x_test, y_test