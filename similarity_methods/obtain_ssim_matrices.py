import cv2, numpy as np, h5py, os
from skimage.metrics import structural_similarity
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

# load all real images into memory
real_images = []
for real_idx in tqdm(range(len(real_names))):
    real_img = cv2.imread(os.path.join(real_folder, real_names[real_idx]))
    real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2GRAY)
    real_images.append(real_img)
real_images = np.asarray(real_images)

# load all generated images into memory
gen_images = []
for gen_idx in tqdm(range(len(generated_names))):
    gen_img = cv2.imread(os.path.join(generated_folder, generated_names[gen_idx]))
    gen_img = cv2.cvtColor(gen_img, cv2.COLOR_BGR2GRAY)
    gen_images.append(gen_img)
gen_images = np.asarray(gen_images)

def build_matrix_ssim(images_1, images_2, same_feat=False):
  matrix = np.zeros((len(images_1), len(images_2)))
  for i in tqdm(range(len(images_1))):
    for j in range(len(images_2)):
      matrix[i][j] = structural_similarity(images_1[i], images_2[j], full=True)[0]
    if same_feat:
      matrix[i][i] = np.average(matrix[i])
  return matrix

gen_real_matrix = build_matrix_ssim(gen_images, real_images)
real_matrix = build_matrix_ssim(real_images, real_images, same_feat=True)
gen_matrix = build_matrix_ssim(gen_images, gen_images, same_feat=True)


print('GENERATED DATASET - REAL DATASET')
print('Average SSIM: ' + str(np.average(gen_real_matrix)))
print('Minimum SSIM: ' + str(np.amin(gen_real_matrix)))
print('Maximum SSIM: ' + str(np.amax(gen_real_matrix)))
print('\nREAL DATASET - REAL DATASET')
print('Average SSIM: ' + str(np.average(real_matrix))) 
print('Minimum SSIM: ' + str(np.amin(real_matrix)))
print('Maximum SSIM: ' + str(np.amax(real_matrix)))
print('\nGENERATED DATASET - GENERATED DATASET')
print('Average SSIM: ' + str(np.average(gen_matrix)))
print('Minimum SSIM: ' + str(np.amin(gen_matrix)))
print('Maximum SSIM: ' + str(np.amax(gen_matrix)))


hf = h5py.File(save_file, 'w')  # open the file in append mode

hf.create_dataset('gen_real_matrix', data=gen_real_matrix)
hf.create_dataset('real_matrix', data=real_matrix)
hf.create_dataset('gen_matrix', data=gen_matrix)
hf.close()
