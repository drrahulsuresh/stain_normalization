import numpy as np
from skimage import io
import os

def rgb_to_od(rgb_image):
    od = -np.log((rgb_image.astype(np.float32) + 1) / 255)
    return od
def od_to_rgb(od_image):
    rgb_image = np.clip(255 * np.exp(-od_image), 0, 255).astype(np.uint8)
    return rgb_image
def get_stain_matrix(od_image, tol=0.15):
    non_bg_pixels = np.all(od_image < 1.0, axis=2)  
    od_image_filtered = od_image[non_bg_pixels]  
    _, _, vh = np.linalg.svd(od_image_filtered.reshape((-1, 3)), full_matrices=False)
    stain_matrix = vh[:2].T
    if stain_matrix[0, 0] < 0: stain_matrix[:, 0] *= -1
    if stain_matrix[0, 1] < 0: stain_matrix[:, 1] *= -1
    return stain_matrix

def stain_normalize(image, reference_image, tol=0.15):
    od_image = rgb_to_od(image)
    od_reference = rgb_to_od(reference_image)
    stain_matrix = get_stain_matrix(od_image, tol)
    reference_stain_matrix = get_stain_matrix(od_reference, tol)
    non_bg_pixels = np.all(od_image < 1.0, axis=2)  
    concentrations = np.linalg.lstsq(stain_matrix, od_image[non_bg_pixels].reshape((-1, 3)).T, rcond=None)[0]
    reference_concentrations = np.linalg.lstsq(reference_stain_matrix, od_reference.reshape((-1, 3)).T, rcond=None)[0]
    
    norm_concentrations = reference_concentrations.mean(axis=1) / concentrations.mean(axis=1)
    concentrations *= norm_concentrations[:, np.newaxis]
    norm_od_image = np.ones_like(od_image)  
    norm_od_image[non_bg_pixels] = np.dot(reference_stain_matrix, concentrations).T
    norm_image = od_to_rgb(norm_od_image)
    norm_image[np.all(image == [0, 0, 0], axis=2)] = [0, 0, 0]  
    return norm_image

image_folder = r"D:\TUD\CC_2nd\h_e\background_removed_images"
reference_image_path = r"D:\TUD\CC_2nd\h_e\images_renamed\031_03.tiff"
output_folder = r"D:\TUD\CC_2nd\h_e\normalized_images_after_bg_removal"

reference_image = io.imread(reference_image_path)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for image_file in os.listdir(image_folder):
    if image_file.endswith(".tiff"):
        image_path = os.path.join(image_folder, image_file)
        image = io.imread(image_path)
        
        normalized_image = stain_normalize(image, reference_image)
        output_path = os.path.join(output_folder, image_file)
        io.imsave(output_path, normalized_image)
        print(f"Normalized and saved: {output_path}")
