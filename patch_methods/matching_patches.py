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
parser.add_argument('--latent_dim', type=int, default=128, help="Dimensions of the latent representations.")
parser.add_argument('--batch_size', type=int, default=16, help="Batch size during training.")
parser.add_argument('--epochs', type=int, default=800, help="Number of epochs to train the network.")
parser.add_argument('--save_file', type=str, default='weights.h5', help="Name of the file where the model weights are/will be saved.")
parser.add_argument('--infer', action='store_true', help="Load model weights and generate submission file.")
parser.add_argument('--submission_file', type=str, default='submission.csv', help="Filename for the submission file.")
args = parser.parse_args()

generated_folder = args.gen_dir
real_folder = args.real_dir
latent_dim = args.latent_dim
batch_size = args.batch_size
epochs = args.epochs
save_file = args.save_file
infer = args.infer
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

def get_patch(img):
    img = tensorflow.image.random_crop(img, [tensorflow.shape(img)[0], crop_shape[0], crop_shape[1], crop_shape[2]])
    return img

feature_extractor = model_feature_extractor()
classifier = model_classifier()

d0 = Input((input_shape))
d1 = Input((input_shape))
d2 = Input((input_shape))
h0 = Lambda(augment, output_shape=(crop_shape,))(d0)
h1 = Lambda(augment, output_shape=(crop_shape,))(d1)
h2 = Lambda(augment, output_shape=(crop_shape,))(d2)
h0 = GaussianNoise(0.1)(h0)
h1 = GaussianNoise(0.1)(h1)
h2 = GaussianNoise(0.1)(h2)
features0 = feature_extractor(h0)
features1 = feature_extractor(h1)
features2 = feature_extractor(h2)

model = Model([d0, d1, d2], [features0, features1, features2])

triplet_loss = mean_squared_error(features0, features1) + \
               K.maximum(0.5 - mean_squared_error(features0, features2), 0)
model.add_loss(triplet_loss)
model.compile(loss=None, optimizer=opt)

# train model

def train_model(epochs, images_to_train):
    batch_count = np.ceil(images_to_train.shape[0] / batch_size)
    for ee in range(1, epochs + 1):
        print('-' * 15, 'Epoch %d' % ee, '-' * 15)
        m_acc = []
        m_loss = []
        for e in tqdm(range(int(batch_count))):
            idx = range(e * batch_size, e * batch_size + batch_size)
            if e * batch_size + batch_size > len(images_to_train):
                idx = range(e * batch_size, len(images_to_train))
            idx2 = np.random.choice(range(len(images_to_train)), size=len(idx))
            res = model.train_on_batch([images_to_train[idx], images_to_train[idx], images_to_train[idx2]], y=None)
            m_loss.append(res)
        print('Loss:' + str(np.average(m_loss)))

if not infer:
    train_model(800, real_images)
    feature_extractor.save_weights(save_file)
else:
    feature_extractor.load_weights(save_file)

crop_real_images = K.eval(get_patch(K.constant(real_images)))
crop_real_imgs2 = K.eval(get_patch(K.constant(real_images)))
crop_gen_imgs = K.eval(get_patch(K.constant(gen_images)))
real_image_feat = feature_extractor.predict(crop_real_images)
real_image_feat2 = feature_extractor.predict(crop_real_imgs2)
gen_image_feat = feature_extractor.predict(crop_gen_imgs)

mse_same = K.eval(mean_squared_error(K.constant(real_image_feat), K.constant(real_image_feat2)))
idx1 = np.tile(np.arange(0, len(real_image_feat)), len(real_image_feat)-1)
idx2 = np.repeat(np.arange(0, len(real_image_feat)), len(real_image_feat))
idx_to_remove = np.arange(0, len(real_image_feat)) * len(real_image_feat)
idx2 = np.delete(idx2, idx_to_remove)
mse_diff = K.eval(mean_squared_error(K.constant(real_image_feat[idx1]),K.constant(real_image_feat[idx2])))
print('Average MSE on the same image:' + str(np.average(mse_same)))
print('Max MSE on the same image:' + str(np.amax(mse_same)))
print('Min MSE on the same image:' + str(np.amin(mse_same)))
print('\nAverage MSE on different images:' + str(np.average(mse_diff)))
print('Max MSE on different images:' + str(np.amax(mse_diff)))
print('Min MSE on different images:' + str(np.amin(mse_diff)))

threshold = np.average(mse_same) - np.std(mse_same)
pred = np.zeros((len(real_images),))
for i in range(len(real_image_feat)):
    gen_img_ft = np.repeat(np.asarray([real_image_feat[i]]), len(gen_image_feat), axis=0)
    mse = K.eval(mean_squared_error(K.constant(gen_img_ft), K.constant(gen_image_feat)))
    if np.amin(mse) < threshold:
      pred[i] = 1

pred = pred.astype(int)

names = real_names
names = [x.split('.')[0] for x in names]

eval_set = dict()
eval_set["id"] = names
eval_set["pred"] = pred

evaluation_df = pd.DataFrame(data=eval_set)
evaluation_df.to_csv(submission_file, sep=',', index=False, header=False)
