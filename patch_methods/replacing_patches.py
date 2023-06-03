import tensorflow, numpy as np, cv2, os, argparse, pandas as pd
from tensorflow.keras.layers import Conv2D, Lambda, Input, Reshape, Dense, concatenate, GaussianNoise, GlobalAveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, UpSampling2D, LeakyReLU
from tensorflow.keras.losses import mean_squared_error, binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tqdm import tqdm
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument('--gen_dir', type=str, default='generated_1', help="Directory of the generated data.")
parser.add_argument('--real_dir', type=str, default='real_unknown_1', help="Directory of the real data.")
parser.add_argument('--weights_file', type=str, default='weights.h5', help="Name of the file where the autoencoder weights are saved.")
parser.add_argument('--submission_file', type=str, default='submission.csv', help="Filename for the submission file.")
args = parser.parse_args()

generated_folder = args.gen_dir
real_folder = args.real_dir
weights_file = args.weights_file
submission_file = args.submission_file

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
crop_shape = (128, 128, 1)
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

def model_feature_extractor():
    d0 = Input((crop_shape))

    h = Conv2D(latent_dim // 8, (3, 3), strides=(2, 2), padding='same')(d0)
    h = LeakyReLU(0.2)(h)
    
    h = conv_block(h, latent_dim // 4, resample='downsample')
    h = conv_block(h, latent_dim // 2, resample='downsample')    
    h = conv_block(h, latent_dim, resample='downsample')    
    h = conv_block(h, latent_dim * 2, resample='downsample')

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
    
    h = res_block(h, units // 2, resample='upsample')
    h = res_block(h, units // 4, resample='upsample')    
    h = res_block(h, units // 8, resample='upsample')    
    h = res_block(h, units // 16, resample='upsample')
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

model.load_weights(weights_file)

# build reconstruction error matrices

real_rec_error_matrix = np.zeros((len(real_images), len(real_images)))
for i in range(len(real_images)):
    # replace patch in real images
    patches = real_images[:, 128:, 64:192]
    new_real_images = np.repeat(np.asarray([real_images[i]]), len(real_images), axis=0)
    new_real_images[:, 128:, 64:192] = patches
    # calculate reconstruction error
    pred = model.predict(new_real_images)
    real_rec_error_matrix[i] = np.average(np.average(K.eval(mean_squared_error(K.constant(np.repeat(np.asarray([real_images[i]]), len(real_images), axis=0)), K.constant(pred))), axis=1), axis=1)
    real_rec_error_matrix[i][i] = np.average(real_rec_error_matrix[i])

print('Replacing patches of real images with those of other real images:')
print('Average reconstruction error: ' + str(np.average(real_rec_error_matrix)))
print('Max reconstruction error: ' + str(np.amax(real_rec_error_matrix)))
print('Min reconstruction error: ' + str(np.amin(real_rec_error_matrix)))

rec_error_matrix = np.zeros((len(real_images), len(gen_images)))
for i in range(len(real_images)):
    # replace patch in real images
    patches = gen_images[:, 128:, 64:192]
    new_real_images = np.repeat(np.asarray([real_images[i]]), len(gen_images), axis=0)
    new_real_images[:, 128:, 64:192] = patches
    # calculate reconstruction error
    pred = model.predict(new_real_images)
    rec_error_matrix[i] = np.average(np.average(K.eval(mean_squared_error(K.constant(np.repeat(np.asarray([real_images[i]]), len(gen_images), axis=0)), K.constant(pred))), axis=1), axis=1)

print('Replacing patches of real images with those of generated images:')
print('Average reconstruction error: ' + str(np.average(rec_error_matrix)))
print('Max reconstruction error: ' + str(np.amax(rec_error_matrix)))
print('Min reconstruction error: ' + str(np.amin(rec_error_matrix)))

threshold = np.average(real_rec_error_matrix) - 2*np.std(real_rec_error_matrix)
thresholding_matrix = np.where(rec_error_matrix < threshold, 1, 0)
sum_thresholding_matrix = np.sum(thresholding_matrix, axis=-1)
pred = np.where(sum_thresholding_matrix > 0, 1, 0)

names = real_names
names = [x.split('.')[0] for x in names]

eval_set = dict()
eval_set["id"] = names
eval_set["pred"] = pred

evaluation_df = pd.DataFrame(data=eval_set)
evaluation_df.to_csv(submission_file, sep=',', index=False, header=False)
