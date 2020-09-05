from process import train_val_test_split

import argparse
import matplotlib.pyplot as plt

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv2D , MaxPooling2D, Flatten
from keras.layers import Dense, BatchNormalization, Activation, Dropout
import keras.backend


def make_model():
    '''CNN architecture'''
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(48,48,1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Conv2D(96, (3, 3), padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    
    model.add(Conv2D(160, (3, 3), padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(7 , activation='sigmoid'))
    
    return model


def plot(history):
    ''' Training vs validation accuracy and loss plots.'''
    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['train', 'val'], loc='upper left')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.savefig('weights/acc_plot.png')
    plt.show()
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['train', 'val'], loc='upper right')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig('weights/loss_plot.png')
    plt.show()

    
def main():
    
    x_train, y_train, x_val, y_val, x_test, y_test = train_val_test_split(args.path)
    print('Shape(x_train, x_val, x_test) - ', x_train.shape, x_val.shape, x_test.shape)
    print('Shape(y_train, y_val, y_test) - ', y_train.shape, y_val.shape, y_test.shape, '\n')
    
    model = make_model()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print('\n==========================Model Summary==========================')
    print(model.summary(), '\n')
    
    history = model.fit(x = x_train, y = y_train, batch_size = 64, epochs = args.epochs, validation_data = (x_val, y_val))
    plot(history)
    
    loss, acc = model.evaluate(x_test, y_test, verbose = 0)
    print('\nAccuracy on test set - ', round(acc*100, 2), '%')
    print('Loss on test set - ', round(loss, 2))
    
    model.save('weights/saved_model.h5')
    print('\nModel and plots saved to weights folder.')
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='data/fer2013.csv', help="path to 'fer2013.csv' file")
    parser.add_argument('--epochs', type=int, default=15, help='number of epochs')
    args = parser.parse_args()
    main()