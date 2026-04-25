import cv2
import numpy as np
from skimage.measure import moments_hu
from skimage.feature import local_binary_pattern

def extract_dct_features(image):
    """
    Extract DCT (Discrete Cosine Transform) features
    Captures frequency-domain characteristics important for forgery detection
    
    Args:
        image: numpy array of shape (H, W, 3) with values in [0, 1]
    
    Returns:
        dct_features: flattened 4x4 DCT coefficients (16 features)
    """
    try:
        # Convert RGB to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        # Apply DCT
        dct = cv2.dct(np.float32(gray))
        
        # Extract top-left 4x4 region (contains most energy)
        dct_features = dct[:4, :4].flatten()
        
        # Normalize
        dct_features = dct_features / (np.linalg.norm(dct_features) + 1e-8)
        
        return dct_features
    except Exception as e:
        print(f"Error in DCT extraction: {e}")
        return np.zeros(16)

def extract_zernike_features(image):
    """
    Extract Zernike moments (shape descriptors)
    Captures shape characteristics that vary between authentic and forged regions
    
    Args:
        image: numpy array of shape (H, W, 3) with values in [0, 1]
    
    Returns:
        zernike_features: Hu moments (7 features)
    """
    try:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        # Normalize
        gray = gray / 255.0
        
        # Compute Hu moments (7 moments)
        moments = moments_hu(gray)
        
        # Hu moments are already normalized but can be unstable
        # Apply log transformation for stability
        zernike_features = np.sign(moments) * np.log(np.abs(moments) + 1e-8)
        
        return zernike_features
    except Exception as e:
        print(f"Error in Zernike extraction: {e}")
        return np.zeros(7)

def extract_lbp_features(image, P=8, R=1):
    """
    Extract Local Binary Pattern (LBP) features
    Captures local texture patterns, very discriminative for forgery detection
    
    Args:
        image: numpy array of shape (H, W, 3) with values in [0, 1]
        P: number of neighboring pixels
        R: radius
    
    Returns:
        lbp_features: histogram of uniform LBP patterns (16 features)
    """
    try:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        # Compute LBP with uniform patterns
        lbp = local_binary_pattern(gray, P=P, R=R, method='uniform')
        
        # Extract histogram of the first 16 bins (uniform patterns + one additional)
        hist, _ = np.histogram(lbp, bins=16, range=(0, 16))
        
        # Normalize histogram
        hist = hist / (np.sum(hist) + 1e-8)
        
        # Return first 16 features
        lbp_features = hist[:16]
        
        return lbp_features
    except Exception as e:
        print(f"Error in LBP extraction: {e}")
        return np.zeros(16)

def extract_all_handcrafted_features(image):
    """
    Extract all handcrafted features (DCT + Zernike + LBP)
    Total: 16 + 7 + 16 = 39 features
    
    Args:
        image: numpy array of shape (H, W, 3) with values in [0, 1]
    
    Returns:
        features: concatenated feature vector (39 features)
    """
    dct_feat = extract_dct_features(image)
    zernike_feat = extract_zernike_features(image)
    lbp_feat = extract_lbp_features(image)
    
    return np.concatenate([dct_feat, zernike_feat, lbp_feat])