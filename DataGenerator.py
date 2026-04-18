from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import keras
from tensorflow import keras
import numpy as np

#Data Generator
class DataGenerator(keras.utils.Sequence):
    def __init__(self, x, y, batch_size=64 , dim=(15,10,1), classes=5, window_size=15, window_step=6, shuffle=True):
        self.x = x  
        self.y = y  
        self.batch_size = batch_size
        self.shuffle = shuffle
        #self.min_max_norm = min_max_norm
        self.dim = dim
        self.window_size = window_size
        self.window_step = window_step
        #self.classes = list(range(0,30)) 
        self.classes = list(range(0,30)) 

        #print("Before",np.unique(self.y))
        LE = LabelEncoder()
        LE.fit(self.classes) 
        self.classes = list(LE.fit_transform(self.classes))
        self.y = LE.transform(self.y)
        #print("After", np.unique(self.y)) 

        self.indexes = np.arange(len(self.x))
        self.__make_segments()
        self.__make_class_index()
        #self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.x_offsets) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        output = self.__data_generation(indexes)
        return output
    
    def __make_segments(self):
        x_offsets = []
        for i in range(len(self.x)):
            for j in range(0, len(self.x[i]) - self.window_size, self.window_step):
                x_offsets.append((i, j))
                #if i< 5:
                    #print(f"Sample {i} | Window starting at {j}")
        
        self.x_offsets = x_offsets
        self.indexes = np.arange(len(self.x_offsets))
        #print(f"\nTotal windows: {len(self.x_offsets)}")
        #print(f"First 20 windows: {self.x_offsets[:20]}")


    def __make_class_index(self):
        
        self.n_classes = len(self.classes)
        self.classes.sort()
        self.class_index = np.zeros(np.max(self.classes)+1, dtype=int) 
        for i, j in enumerate(self.classes):
            self.class_index[j] = i

    
    def __data_generation(self, indexes):
        X = np.empty((self.batch_size, *self.dim))            
        y = np.empty((self.batch_size), dtype=int)

        
        for k, index in enumerate(indexes):
            i, j = self.x_offsets[index]
            
            x = self.x[i][j:j + self.window_size]  

            #print(f"Sample {i} : Window {k}: Starting at index {j}")

            #if self.min_max_norm:
                #max_x = x.max()
                #min_x = x.min()
                #x = (x - min_x) / (max_x - min_x)  

            if np.prod(x.shape) == np.prod(self.dim):
                x = np.reshape(x, self.dim)  
            else:
                raise Exception(f'Generated sample dimension mismatch. Found {x.shape}, expected {self.dim}.')

            X[k, ] = x
            #print("self.indexes[i]:", self.indexes[i])
            #print("self.y shape:", self.y.shape)
            
            stimulus = int(self.y[i])
            mapped_class = self.class_index[stimulus]
            y[k] = mapped_class
            one_hot = to_categorical([mapped_class], num_classes=len(self.classes))[0]
            #print(f"Window {k} | Original stimulus: {stimulus}, Mapped class: {mapped_class}, One-hot: {one_hot}")


        #print(f"Labels before {y}")
        y = to_categorical(y, num_classes=len(self.classes)) 
        #print(f"labels after: {(y)}")
            
        #y = keras.utils.to_categorical(y, num_classes=self.n_classes+1)
        #print(f"One-hot-encoded {y.tolist()}")

        output = (X, y)
        return output

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)