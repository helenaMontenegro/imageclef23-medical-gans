import cv2, numpy as np, h5py, os, tensorflow
from tensorflow.keras.applications.resnet50 import preprocess_input
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gen_dir', type=str, default='generated_1', help="Directory of the generated data.")
parser.add_argument('--real_dir', type=str, default='real_unknown_1', help="Directory of the real data.")
parser.add_argument('--save_file', type=str, default='ssim_matrices.hdf5', help="Filename where the SSIM matrices will be stored.")
args = parser.parse_args()

generated_folder = args.gen_dir
real_folder = args.real_dir
save_file = args.save_file

generated_names = os.listdir(generated_folder)
generated_names.sort()
real_names = os.listdir(real_folder)
real_names.sort()

# load all generated images into memory
gen_images = []
for gen_idx in tqdm(range(len(generated_names))):
    gen_img = cv2.imread(os.path.join(generated_folder, generated_names[gen_idx]))
    gen_img = cv2.cvtColor(gen_img, cv2.COLOR_BGR2RGB)
    gen_img = cv2.resize(gen_img, (224, 224), interpolation = cv2.INTER_AREA)
    gen_images.append(gen_img)
gen_images = np.asarray(gen_images)
gen_images = preprocess_input(gen_images)

# load all real images into memory
real_images = []
for idx in tqdm(range(len(real_names))):
    img = cv2.imread(os.path.join(real_folder, real_names[idx]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)
    real_images.append(img)
real_images = np.asarray(real_images)
real_images = preprocess_input(real_images)

def build_matrix(feat_1, feat_2, same_feat=False):
  matrix = np.zeros((len(feat_1), len(feat_2)))
  for i in tqdm(range(len(feat_1))):
    for j in range(len(feat_2)):
      matrix[i][j] = mean_squared_error(feat_1[i], feat_2[j])
    if same_feat:
      matrix[i][i] = np.average(matrix[i])
  return matrix

model = ResNet50(weights="imagenet", include_top=False, pooling='avg', input_shape=(224, 224, 3))
gen_features = model.predict(gen_images)
real_features = model.predict(real_images)

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
