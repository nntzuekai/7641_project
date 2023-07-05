from EEGModels import EEGNet
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import pickle as p



if __name__ == '__main__':
    
    samples = 260
    channels = 56
    kernels = 1

    with open('all_users.pkl','rb') as f:
        data = p.load(f)
    
    keys=list(data.keys())

    # [key0, key1]=keys[0:2]
    [key0, key1]=np.random.choice(keys, 2, False)

    [X_train, Y_train]=data[key0]
    X_train=X_train.reshape(X_train.shape[0], channels, samples, kernels)

    [X_test0, Y_test0]=data[key1]
    X_test0=X_test0.reshape(X_test0.shape[0], channels, samples, kernels)

    test_idx=np.random.permutation(X_test0.shape[0])

    X_test_to_train=X_test0[test_idx[:10]]
    X_test=X_test0[test_idx[10:]]

    Y_test_to_train=Y_test0[test_idx[:10]]
    Y_test=Y_test0[test_idx[10:]]
    
    X_train=np.concatenate((X_train, X_test_to_train), axis=0)
    Y_train=np.concatenate((Y_train, Y_test_to_train), axis=0)



    model = EEGNet(nb_classes = 1, Chans = channels, Samples = samples)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
    checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint1.h5', verbose=1, save_best_only=True)

    vc = np.unique(Y_train, return_counts=True)[1]
    cw = {0:vc[1]/vc[0], 1:1}

    fittedModel = model.fit(X_train, Y_train, batch_size = 16, epochs = 300, 
        verbose = 2, callbacks=[checkpointer], class_weight = cw)
    
    probs       = model.predict(X_test)
    preds       = (probs.flatten() >= 0.5).astype(int)
    acc         = np.mean(preds == Y_test)
    print("Classification accuracy: %f " % (acc))