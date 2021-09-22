import numpy as np
import matplotlib.pyplot as plt
import pathlib
import time
import tensorflow as tf

test = list(pathlib.Path('data/test').glob('*.jpg'))

def train(model,dataset,epochs):
    for epoch in range(epochs):
        print(f'EPOCH: {epoch}')
        model.fit(dataset)
        example = plt.imread(test[epoch])
        pred = model.predict(example)
        pred = np.broadcast_to(pred[...,None],example.shape)
        merge = np.concatenate([example/255,pred],axis=1)
        plt.imshow(merge)
        plt.show()
print('--train(model,dataset,epochs)')

def convert_test_image(model):
    for t in test:
        example = plt.imread(t)
        start = time.time()
        pred = model.predict(example)
        cost = time.time()-start
        pred = np.broadcast_to(pred[...,None],example.shape)
        merge = np.concatenate([example/255,pred],axis=1)
        plt.imshow(merge)
        plt.title(f'Cost Time: {cost}')
        plt.show()
print('--convert_test_image(model)')

def save_model(model,model_name):
    model.save(pathlib.Path('SavedModels',model_name).as_posix())
print('--save_model(model,model_name)')

def load_model(model_name):
    return tf.keras.models.load_model(pathlib.Path('SavedModels',model_name).as_posix())
print('--load_model(model_name)')