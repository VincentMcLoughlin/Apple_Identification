import numpy as np
from skimage.io import imread_collection

trainPath = "Fruit-Images-Dataset-master/Training"
testPath = "Fruit-Images-Dataset-master/Test"

#train data
braeburn_train = np.array(imread_collection(trainPath + "/Apple Braeburn/*"))
golden1_train = np.array(imread_collection(trainPath + "/Apple Golden 1/*"))
granny_train = np.array(imread_collection(trainPath + "/Apple Granny Smith/*"))
red1_train = np.array(imread_collection(trainPath + "/Apple Red 1/*"))
redDelicious_train = np.array(imread_collection(trainPath + "/Apple Red Yellow 1/*"))
redYellow1_train = np.array(imread_collection(trainPath + "/Apple Red Yellow 1/*"))
tomato_train = np.array(imread_collection(trainPath + "/Tomato 1/*"))
tomato_train = tomato_train[:492,:,:,:]

#Test Data
braeburn_test = np.array(imread_collection(testPath + "/Apple Braeburn/*"))
golden1_test = np.array(imread_collection(testPath + "/Apple Golden 1/*"))
granny_test = np.array(imread_collection(testPath + "/Apple Granny Smith/*"))
red1_test = np.array(imread_collection(testPath + "/Apple Red 1/*"))
redDelicious_test = np.array(imread_collection(testPath + "/Apple Red Yellow 1/*"))
redYellow1_test = np.array(imread_collection(testPath + "/Apple Red Yellow 1/*"))
tomato_test = np.array(imread_collection(testPath + "/Tomato 1/*"))
tomato_test = tomato_test[:164,:,:,:]

#Total 
braeburn_train = np.concatenate((braeburn_train,braeburn_test), 0)
golden1_train = np.concatenate((golden1_train,golden1_test), 0)
granny_train = np.concatenate((granny_train,granny_test), 0)
red1_train = np.concatenate((red1_train,red1_test), 0)
redDelicious_train = np.concatenate((redDelicious_train,redDelicious_test), 0)
redYellow1_train = np.concatenate((redYellow1_train,redYellow1_test), 0)
tomato_train = np.concatenate((tomato_train,tomato_test), 0)


#Label Data
braeburn_train_length = np.ones(shape=(braeburn_train.shape[0],1))*0
golden1_train_length = np.ones(shape=(golden1_train.shape[0],1))*1
granny_train_length = np.ones(shape=(granny_train.shape[0],1))*2
red1_train_length = np.ones(shape=(red1_train.shape[0],1))*3
redDelicious_train_length = np.ones(shape=(redDelicious_train.shape[0],1))*4
redYellow1_train_length =  np.ones(shape=(redYellow1_train.shape[0],1))*5
tomato_train_length = np.ones(shape=(tomato_train.shape[0],1))*6

balanced_train_data = np.concatenate((braeburn_train, golden1_train, granny_train,red1_train,redDelicious_train,
                             redYellow1_train,tomato_train), 0)

balanced_train_labels = np.concatenate((braeburn_train_length, golden1_train_length, granny_train_length,
                                red1_train_length,redDelicious_train_length,redYellow1_train_length, 
                                tomato_train_length),0)

balanced_test_data = np.concatenate((braeburn_test, golden1_test, granny_test,red1_test,redDelicious_test,
                             redYellow1_test,tomato_test), 0)

print(braeburn_train.shape)
print(golden1_train.shape)
print(granny_train.shape)
print(red1_train.shape)
print(redDelicious_train.shape)
print(redYellow1_train.shape)
print(tomato_train.shape)

np.save("total_balanced_train_data",balanced_train_data)
np.save("total_balanced_train_labels",balanced_train_labels)
np.save("total_balanced_test_data",balanced_test_data)