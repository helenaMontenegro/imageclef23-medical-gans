import numpy as np, cv2, os, pandas as pd, argparse, tensorflow, h5py
from tqdm import tqdm
from tensorflow.keras.layers import Conv2D, Input, Reshape, Dense, Add, GlobalAveragePooling2D, BatchNormalization, UpSampling2D, LeakyReLU
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K

parser = argparse.ArgumentParser()
parser.add_argument('--gen_dir', type=str, default='generated_1', help="Directory of the generated data.")
parser.add_argument('--real_dir', type=str, default='real_unknown_1', help="Directory of the real data.")
parser.add_argument('--architecture', type=str, default='simple', help="Architecture of the autoencoder: simple or resnet")
parser.add_argument('--latent_dim', type=int, default=128, help="Dimensions of the latent representations.")
parser.add_argument('--batch_size', type=int, default=32, help="Batch size during training.")
parser.add_argument('--epochs', type=int, default=200, help="Number of epochs to train the network.")
parser.add_argument('--weights_file', type=str, default='weights.h5', help="Name of the file where the model weights are/will be saved.")
parser.add_argument('--infer', action='store_true', help="Load model weights.")
args = parser.parse_args()

generated_folder = args.gen_dir
real_folder = args.real_dir
latent_dim = args.latent_dim
batch_size = args.batch_size
epochs = args.epochs
weights_file = args.weights_file
infer = args.infer
submission_file = args.submission_file
architecture = args.architecture

# load data

generated_names = os.listdir(generated_folder)
generated_names.sort()
real_names = os.listdir(real_folder)
real_names.sort()

gen_images = []
for gen_idx in tqdm(range(len(generated_names))):
    gen_img = cv2.imread(os.path.join(generated_folder, generated_names[gen_idx]))
    gen_img = cv2.cvtColor(gen_img, cv2.COLOR_BGR2GRAY)
    gen_images.append(gen_img)
  
gen_images = np.asarray(gen_images)
gen_images = np.reshape(gen_images, (-1, 256, 256, 1))
gen_images = (gen_images - 127.5) / 127.5

real_images = []
labels = []
for idx in tqdm(range(len(used_names))):
    img = cv2.imread(os.path.join(used_folder, used_names[idx]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    real_images.append(img)
    labels.append(1)

for idx in tqdm(range(len(not_used_names))):
    img = cv2.imread(os.path.join(not_used_folder, not_used_names[idx]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    real_images.append(img)
    labels.append(0)

labels = np.asarray(labels)
real_images = np.asarray(real_images)
real_images = np.reshape(real_images, (-1, 256, 256, 1))
real_images = (real_images - 127.5) / 127.5

# model definition
input_shape = (256, 256, 1)
opt = Adam(learning_rate=1e-4)

def conv_block(x, filters_out, kernel_size=(3, 3), resample=None):
    if resample == None:
        x = Conv2D(filters=filters_out, kernel_size=kernel_size, strides=(1, 1), padding="same", kernel_initializer=glorot_uniform(seed=0))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
    elif resample == 'downsample':
        x = Conv2D(filters=filters_out, kernel_size=kernel_size, strides=(2, 2), padding="same", kernel_initializer=glorot_uniform(seed=0))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
    elif resample == 'upsample':
        x = UpSampling2D()(x)
        x = Conv2D(filters=filters_out, kernel_size=kernel_size, strides=(1, 1), padding="same", kernel_initializer=glorot_uniform(seed=0))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
    return x   

def res_block(x, filters_out, kernel_size=(3, 3), resample=None):
    if resample == None:
        shortcut = x
        x = Conv2D(filters=filters_out, kernel_size=kernel_size, strides=(1, 1), padding="same", kernel_initializer=glorot_uniform(seed=0))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(filters=filters_out, kernel_size=kernel_size, strides=(1, 1), padding="same", kernel_initializer=glorot_uniform(seed=0))(x)
        x = BatchNormalization()(x)
        x = Add()([x, shortcut])
        x = LeakyReLU(alpha=0.2)(x)
    elif resample == 'downsample':
        shortcut = Conv2D(filters=filters_out, kernel_size=kernel_size, strides=(2, 2), padding="same", kernel_initializer=glorot_uniform(seed=0))(x)
        shortcut = BatchNormalization()(shortcut)
        x = Conv2D(filters=filters_out, kernel_size=kernel_size, strides=(2, 2), padding="same", kernel_initializer=glorot_uniform(seed=0))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(filters=filters_out, kernel_size=kernel_size, strides=(1, 1), padding="same", kernel_initializer=glorot_uniform(seed=0))(x) 
        x = BatchNormalization()(x)
        x = Add()([x, shortcut])
        x = LeakyReLU(alpha=0.2)(x)
    elif resample == 'upsample':
        shortcut = UpSampling2D()(x)
        shortcut = Conv2D(filters=filters_out, kernel_size=kernel_size, strides=(1, 1), padding="same", kernel_initializer=glorot_uniform(seed=0))(shortcut)
        shortcut = BatchNormalization()(shortcut)
        x = UpSampling2D()(x)
        x = Conv2D(filters=filters_out, kernel_size=kernel_size, strides=(1, 1), padding="same", kernel_initializer=glorot_uniform(seed=0))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(filters=filters_out, kernel_size=kernel_size, strides=(1, 1), padding="same", kernel_initializer=glorot_uniform(seed=0))(x)
        x = BatchNormalization()(x)
        x = Add()([x, shortcut])
        x = LeakyReLU(alpha=0.2)(x)
    return x  

if architecture == 'simple':
    block_func = conv_block
elif architecture == 'resnet':
    block_func = res_block
else:
    print('There is not architecture named ' + architecture + '. Try: simple or resnet.')

def model_feature_extractor():
    d0 = Input((input_shape))

    h = Conv2D(latent_dim // 8, (3, 3), strides=(2, 2), padding='same')(d0)
    h = LeakyReLU(0.2)(h)
    
    h = block_func(h, latent_dim // 4, resample='downsample')
    h = block_func(h, latent_dim // 2, resample='downsample')    
    h = block_func(h, latent_dim, resample='downsample')    
    h = block_func(h, latent_dim * 2, resample='downsample')

    h1 = GlobalAveragePooling2D()(h)
    features = Dense(latent_dim, name="medical_features")(h1)

    feature_extractor = Model(d0, features)
    feature_extractor.compile(loss='binary_crossentropy', optimizer=opt)

    return feature_extractor

def model_generator():
    units = 256
    input_vector = Input(shape=(latent_dim,))
    h = Dense(input_shape[0]//16 * input_shape[1]//16 * units)(input_vector)
    h = LeakyReLU(0.2)(h)
    h = Reshape((input_shape[0]//16, input_shape[1]//16, units))(h)
    
    h = block_func(h, units // 2, resample='upsample')
    h = block_func(h, units // 4, resample='upsample')    
    h = block_func(h, units // 8, resample='upsample')    
    h = block_func(h, units // 16, resample='upsample')
    h = Conv2D(1, (3, 3), strides=(1, 1), padding='same', activation='tanh')(h)  # 8*6*64
    
    generator = Model(input_vector, h, name="Generator")
    generator.compile(loss='binary_crossentropy', optimizer=opt)

    return generator

feature_extractor = model_feature_extractor()
generator = model_generator()

d0 = Input((input_shape))
features = feature_extractor(d0)
reconstruction = generator(features)

model = Model(d0, reconstruction)
model.compile(loss='mean_squared_error', optimizer=opt)


if not infer:
    model.fit(x=images, y=images, batch_size=batch_size, epochs=epochs)
    model.save_weights(weights_file)
else:
    model.load_weights(weights_file)

gen_real_matrix = 1 - build_matrix(gen_features, real_features)
real_matrix = 1 - build_matrix(real_features, real_features, same_feat=True)
gen_matrix = 1 - build_matrix(gen_features, gen_features, same_feat=True)

print('GENERATED DATASET - REAL DATASET')
print('Average similarity: ' + str(np.average(gen_real_matrix)))
print('Minimum similarity: ' + str(np.amin(gen_real_matrix)))
print('Maximum similarity: ' + str(np.amax(gen_real_matrix)))
print('\nREAL DATASET - REAL DATASET')
print('Average similarity: ' + str(np.average(real_matrix))) 
print('Minimum similarity: ' + str(np.amin(real_matrix)))
print('Maximum similarity: ' + str(np.amax(real_matrix)))
print('\nGENERATED DATASET - GENERATED DATASET')
print('Average similarity: ' + str(np.average(gen_matrix)))
print('Minimum similarity: ' + str(np.amin(gen_matrix)))
print('Maximum similarity: ' + str(np.amax(gen_matrix)))

hf = h5py.File(save_file, 'w')  # open the file in append mode

hf.create_dataset('gen_real_matrix', data=gen_real_matrix)
hf.create_dataset('real_matrix', data=real_matrix)
hf.create_dataset('gen_matrix', data=gen_matrix)
hf.close()