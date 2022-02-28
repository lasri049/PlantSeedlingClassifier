import random
import os
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util
from skimage import io
import numpy as np



def random_rotation(image_array: ndarray):
    #pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25,25)
    return sk.transform.rotate(image_array, random_degree)

def random_noise(image_array: ndarray):
    #add random noise to the image
    return sk.util.random_noise(image_array, mode='gaussian', seed=13, var=0.005)

def horizontal_flip(image_array: ndarray):
    #horizontally flip image
    return image_array[:,::-1]

def random_translation(image_array: ndarray):
    random_rot=random.uniform(-10,10)
    random_shear=random.uniform(-1,1)
    shape=image_array.shape
    shape_size=shape[:2]
    center=np.float32(shape_size) / 2. - 0.5
    
    pre = transform.SimilarityTransform(translation=-center)
    affine = transform.AffineTransform(rotation=random_rot, shear=random_shear, translation=center)
    tf_img = pre + affine
    
    return transform.warp(image_array, tf_img.params, mode='reflect')
    

def data_generator(folder_path):
    
    avlbl_trsfm = {'rotate' : random_rotation,
            'noise' : random_noise,
            'horizontal_flip' : horizontal_flip,
            'translation' : random_translation}
    
    no_files = 0
    
    for f in os.listdir(folder_path):
        
           images = os.path.join(folder_path, f)
        
           
        
           no_files = 800 - len(os.listdir(images))
           
           #print('no_files is ', no_files)
           
           
           
           num_gen_files = 0
           
           while num_gen_files <= no_files:
               #print("while1")
               
               image_path = os.path.join(images, random.choice(os.listdir(images)))
               
               image_to_trsfm = sk.io.imread(image_path)
               
               # random no of transformations to be applied
               num_trsfm_app = random.randint(1, len(avlbl_trsfm))
               
               num_trsfm = 0
               
               trsfmd_image = None
               
               while num_trsfm <= num_trsfm_app:
                   #print('while2')
                   #random transformation to be applied for a single image
                   key = random.choice(list(avlbl_trsfm))
                   
                   trsfmd_image = avlbl_trsfm[key](image_to_trsfm)
                   
                   num_trsfm += 1
            
               new_file_path = '%s/augmented_img_%s.jpg' % (images, num_gen_files)
               io.imsave(new_file_path, trsfmd_image)       
               
               num_gen_files += 1
           
           print("Augmentation for %s images is complete" % (f))
        