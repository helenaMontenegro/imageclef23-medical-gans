# Evaluating Privacy on Synthetic Images Generated using GANs: Contributions of the VCMI Team to ImageCLEFmedical GANs 2023

This is the official repository for the [VCMI](https://vcmi.inesctec.pt/)'s team submission to [ImageCLEFmedical GANs 2023](https://www.imageclef.org/2023/medical/gans).

**Task Goal**: Evaluate whether synthetic medical images generated using deep generative models have identifiable properties of the training data, threatening its privacy.

**Task Description**: Given a set of real images and a set of synthetic images generated using a generative adversarial network (GAN), classify the real images according to whether they were used in the training of the GAN.

**Proposed Methods**: 
* Similarity-based methods, which measure the similarity between real and generated images and use it to classify the real images.
* Autoencoder-based methods, which use autoencoders to classify the real images.
* Patch-based methods, which extract patches from images and use them for classification.


## Requirements
* tensorflow (version: 2.12.0)
* scikit-learn (version: 1.2.1)
* scikit-image (version: 0.19.3)
* h5py (version: 3.1.0)

## Similarity-based methods

To apply the similarity-based methods, first we need to obtain similarity matrices containing the similarity between the images. 
Two methods are available on the ```similarity_methods``` folder to calculate the similarity between the images: 
* Using Structural Similarity Index Measure (```obtain_ssim_matrices.py```)
* Calculating the distance between the latent representations of the images, obtained with a pre-trained ResNet-50 network (```obtain_resnet_matrices.py```)

These scripts are executed using ```python obtain_ssim_matrices.py``` or ```python obtain_resnet_matrices.py```, and, optionally, with the following parameters:

Name | Type | Default | Description
--- | --- | --- | ---
--gen_dir | string | generated_1 | Path to the directory that contains the generated images
--real_dir | string | real_unknown_1 | Path to the directory that contains the real images
--save_file | string | ssim_matrices.hdf5 | Name of the hdf5 file where the matrices will be saved

These scripts generate an hdf5 file that contains a matrix with the similarity between real images (```real_matrix```), a matrix with the similarity between real and generated images (```gen_real_matrix```), and a matrix with the similarity between generated images (```gen_matrix```).
To apply the similarity methods to the matrices, we can run the script ```python similarity_methods.py``` with the following parameters:

Name | Type | Default | Description
--- | --- | --- | ---
--infer | bool | False | Generates submission file if True
--method | string | all | Defines which similarity method to apply: all, threshold_max, threshold_avg, retrieval, ranking, clustering or ensemble
--matrix_file | string | ssim_matrices.hdf5 | Path to the file that holds the similarity matrices
--submission_file | string | submission.csv | Name of the file where the results of the selected method will be saved
--real_dir | string | real_unknown_1 | Path to the directory that contains the real images (required on inference)

The final results will be available on the file defined using the argument: ```--submission_file```.

## Autoencoder-based methods

The autoencoder-based methods are available on the ```autoencoder_methods``` folder.

### Outlier detection with autoencoder trained on generated data

Run ```python basic_autoencoder.py``` with the following parameters:

Name | Type | Default | Description
--- | --- | --- | ---
--gen_dir | string | generated_1 | Path to the directory that contains the generated images
--real_dir | string | real_unknown_1 | Path to the directory that contains the real images
--architecture | string | simple | Architecture of the autoencoder: simple or resnet
--latent_dim | int | 128 | Dimensions of the latent representations of the autoencoder
--batch_size | int | 32 | Batch size during training
--epochs | int | 200 | Number of epochs to train the network
--weights_file | string | weights.hdf5 | Path to the file  where the model weights are/will be saved
--infer | bool | False | Loads model weights and generates submission file
--submission_file | string | submission.csv | Name of the file where the results of the selected method will be saved

The final results will be available on the file defined using the argument: ```--submission_file```.

### Autoencoder trained on real and generated data

Run ```python autoencoder_similarity.py``` with the following parameters:

Name | Type | Default | Description
--- | --- | --- | ---
--gen_dir | string | generated_1 | Path to the directory that contains the generated images
--real_dir | string | real_unknown_1 | Path to the directory that contains the real images
--architecture | string | simple | Architecture of the autoencoder: simple or resnet
--latent_dim | int | 128 | Dimensions of the latent representations of the autoencoder
--batch_size | int | 32 | Batch size during training
--epochs | int | 200 | Number of epochs to train the network
--weights_file | string | weights.hdf5 | Path to the file  where the model weights are/will be saved
--infer | bool | False | Loads model weights and generates submission file
--matrix_file | string | sim_matrices.hdf5 | Path to the file where the similarity matrices will be stored

This script generates an hdf5 file that contains a matrix with the similarity between images, which can be provided to the ```similarity_methods.py``` script.

### Autoencoder with two decoders

Run ```python 2_decoder_autoencoder.py``` with the following parameters:

Name | Type | Default | Description
--- | --- | --- | ---
--gen_dir | string | generated_1 | Path to the directory that contains the generated images
--real_dir | string | real_unknown_1 | Path to the directory that contains the real images
--architecture | string | simple | Architecture of the autoencoder: simple or resnet
--latent_dim | int | 128 | Dimensions of the latent representations of the autoencoder
--batch_size | int | 32 | Batch size during training
--epochs | int | 200 | Number of epochs to train the network
--weights_file | string | weights.hdf5 | Path to the file  where the model weights are/will be saved
--infer | bool | False | Loads model weights and generates submission file
--submission_file | string | submission.csv | Name of the file where the results of the selected method will be saved
--generate_matrix | bool | False | Generates similarity matrix to apply similarity-based methods
--matrix_file | string | sim_matrices.hdf5 | Path to the file where the similarity matrices will be stored

This script (with the parameter ```--generate_matrix``` set) generates an hdf5 file that contains a matrix with the similarity between images, which can be provided to the ```similarity_methods.py``` script. It also produces outlier detection results that will be available on the file defined using the argument: ```--submission_file```.

## Patch-based methods

The patch-based methods are available on the ```patch_methods``` folder.

### Matching Patches

Run ```python matching_patches.py``` with the following parameters:

Name | Type | Default | Description
--- | --- | --- | ---
--gen_dir | string | generated_1 | Path to the directory that contains the generated images
--real_dir | string | real_unknown_1 | Path to the directory that contains the real images
--latent_dim | int | 128 | Dimensions of the latent representations of the autoencoder
--batch_size | int | 16 | Batch size during training
--epochs | int | 800 | Number of epochs to train the network
--save_file | string | weights.hdf5 | Path to the file  where the model weights are/will be saved
--infer | bool | False | Loads model weights and generates submission file
--submission_file | string | submission.csv | Name of the file where the results of the selected method will be saved

The final results will be available on the file defined using the argument: ```--submission_file```.

### Replacing Patches

Run ```python replacing_patches.py``` with the following parameters:

Name | Type | Default | Description
--- | --- | --- | ---
--gen_dir | string | generated_1 | Path to the directory that contains the generated images
--real_dir | string | real_unknown_1 | Path to the directory that contains the real images
--weights_file | string | weights.hdf5 | Name of the file where the autoencoder weights are saved
--submission_file | string | submission.csv | Name of the file where the results of the selected method will be saved

The final results will be available on the file defined using the argument: ```--submission_file```.
