import os
import numpy as np
import imageio
import rawpy
from skimage import filters
from sklearn.decomposition import PCA, FastICA
import rasterio
from rasterio.transform import from_origin
import warnings
from rasterio.errors import NotGeoreferencedWarning

# Suppress NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

def load_image(file_path):
    """Load an image from a file path depending on the file extension."""
    file_extension = file_path.split('.')[-1].lower()
    if file_extension in ['jpeg', 'jpg']:
        return imageio.imread(file_path)
    elif file_extension == 'tif':
        return imageio.imread(file_path)
    elif file_extension == 'dng':
        with rawpy.imread(file_path) as raw:
            return raw.postprocess()
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

def load_multispectral_image_bands(input_folder):
    """Load separate band images from the specified folder and combine them into a single array."""
    band_files = [f for f in os.listdir(input_folder) if f.split('.')[-1].lower() in ['tif', 'jpeg', 'jpg', 'dng']]
    band_files.sort()  # Ensure the bands are in the correct order
    bands = [load_image(os.path.join(input_folder, band_file)) for band_file in band_files]
    return np.stack(bands, axis=0)

def perform_pca(image):
    """Perform Principal Component Analysis (PCA) on the image."""
    pca = PCA(n_components=image.shape[0])
    return pca.fit_transform(image.reshape(image.shape[0], -1).T).T.reshape(image.shape)

def perform_ica(image):
    """Perform Independent Component Analysis (ICA) on the image."""
    ica = FastICA(n_components=image.shape[0])
    return ica.fit_transform(image.reshape(image.shape[0], -1).T).T.reshape(image.shape)

def create_image_ratio(image, band1, band2):
    """Calculate the ratio of two bands in the image."""
    epsilon = 1e-10  # Avoid division by zero
    denominator = image[band2] + epsilon
    ratio = np.divide(image[band1], denominator, where=denominator != 0)
    return ratio

def apply_gaussian_blur(image, sigma=1):
    """Apply Gaussian blur to the image."""
    return filters.gaussian(image, sigma=sigma)

def perform_mnf(image):
    """Perform Minimum Noise Fraction (MNF) transformation on the image."""
    return image  # Placeholder for MNF transformation

def save_image(image, output_folder, file_name, process_name):
    """Save each band of the image as a separate 8-bit TIFF file with descriptive filenames."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for band_index in range(image.shape[0]):
        # Normalize image to 8-bit range [0, 255]
        band = image[band_index]
        band_min, band_max = np.min(band), np.max(band)
        band_8bit = ((band - band_min) / (band_max - band_min) * 255).astype(np.uint8)

        filename = os.path.join(output_folder, f"{file_name}_{process_name}_band{band_index+1}.tif")
        with rasterio.open(
            filename, 'w',
            driver='GTiff',
            height=band_8bit.shape[0],
            width=band_8bit.shape[1],
            count=1,  # Saving one band per file
            dtype=band_8bit.dtype
        ) as dst:
            dst.write(band_8bit, 1)  # Write this band as the first (and only) band in the TIFF

def process_images(input_folder, output_folder):
    """Process all bands in the input folder as parts of a multispectral image."""
    image = load_multispectral_image_bands(input_folder)
    if image.shape[0] < 2:
        print("Not enough bands for certain operations.")
        return

    # Process and save each type of image
    processes = [
        ('PCA', perform_pca),
        ('ICA', perform_ica),
        ('Ratio_B1_B2', lambda img: create_image_ratio(img, 0, 1)),
        ('Gaussian_Blur', lambda img: apply_gaussian_blur(img[0], sigma=2)),
        ('MNF', perform_mnf)
    ]

    for process_name, func in processes:
        processed_image = func(image)
        save_image(processed_image, output_folder, 'multispectral', process_name)


# Usage example
input_folder = 'F:\\Isadore\\IsadoreFlyLeafD\\IsadoreFlyLeafd_DataCube Raw Images'
output_folder = 'F:\\Isadore\\IsadoreFlyLeafD\\PythonProcessed'
process_images(input_folder, output_folder)
