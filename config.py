"""
Configuration for Forgery Detection Model
Easily adjust hyperparameters for different scenarios
"""

class Config:
    # ==================== DATASET ====================
    DATASET_PATH = "CoMoFoD_small_v2/"
    IMAGE_SIZE = 256
    TRAIN_SPLIT = 0.70
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    
    # ==================== MODEL ====================
    # Backbone models
    CONVNEXT_MODEL = "convnext_base"      # tiny, small, base, large
    EFFICIENT_MODEL = "efficientnet_b4"   # b0, b1, b2, b3, b4, b5
    VIT_MODEL = "vit_base_patch16_224"    # vit_tiny, vit_small, vit_base, vit_large
    
    # Feature dimensions
    HANDCRAFTED_DIM = 39  # DCT(16) + Zernike(7) + LBP(16)
    
    # ==================== TRAINING ====================
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 1e-4
    
    # Learning rate scheduling
    WARMUP_EPOCHS = 5
    MIN_LR_RATIO = 0.1  # Minimum LR as ratio of initial LR
    
    # ==================== REGULARIZATION ====================
    DROPOUT_RATE = 0.35
    LABEL_SMOOTHING = 0.05  # 0 for authentic, 1 → 0.95 for forged
    GRAD_CLIP_NORM = 1.0
    
    # ==================== OPTIMIZATION ====================
    OPTIMIZER = "AdamW"  # Adam, SGD, AdamW
    BETA1 = 0.9
    BETA2 = 0.999
    
    # Early stopping
    PATIENCE = 15
    BEST_METRIC = "val_loss"  # val_loss or val_acc
    
    # ==================== DATA AUGMENTATION ====================
    # Geometric augmentations
    ROTATION = 20
    TRANSLATE = (0.15, 0.15)
    SCALE = (0.85, 1.15)
    SHEAR = 10
    PERSPECTIVE_DISTORTION = 0.15
    
    # Frequency augmentations
    BLUR_SIGMA = (0.1, 1.5)
    BLUR_KERNEL = 3
    SHARPNESS_FACTOR = 2
    
    # Color augmentations
    BRIGHTNESS = 0.3
    CONTRAST = 0.3
    SATURATION = 0.3
    HUE = 0.08
    
    # Additional augmentations
    RANDOM_INVERT_P = 0.05
    RANDOM_AUTOCONTRAST_P = 0.1
    RANDOM_EQUALIZE_P = 0.1
    
    # Normalization (ImageNet)
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]
    
    # ==================== DEVICE ====================
    DEVICE = "cuda"  # cuda or cpu
    NUM_WORKERS = 4
    PIN_MEMORY = True
    
    # ==================== LOGGING ====================
    SAVE_INTERVAL = 10  # Save checkpoint every N epochs
    LOG_INTERVAL = 100  # Log metrics every N batches
    SAVE_BEST_MODEL = True
    CHECKPOINT_DIR = "./"
    MODEL_NAME = "best_model.pth"


class ConfigTiny:
    """Lightweight config for quick experiments"""
    CONVNEXT_MODEL = "convnext_tiny"
    EFFICIENT_MODEL = "efficientnet_b0"
    BATCH_SIZE = 16
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    DROPOUT_RATE = 0.2
    PATIENCE = 5


class ConfigLarge:
    """High-capacity config for maximum accuracy"""
    CONVNEXT_MODEL = "convnext_large"
    EFFICIENT_MODEL = "efficientnet_b5"
    VIT_MODEL = "vit_large_patch16_224"
    BATCH_SIZE = 64
    EPOCHS = 150
    LEARNING_RATE = 2e-5
    DROPOUT_RATE = 0.4
    WEIGHT_DECAY = 2e-4
    PATIENCE = 20


class ConfigBalanced:
    """Balanced config for 97-98% accuracy (RECOMMENDED)"""
    CONVNEXT_MODEL = "convnext_base"
    EFFICIENT_MODEL = "efficientnet_b4"
    VIT_MODEL = "vit_base_patch16_224"
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 1e-4
    DROPOUT_RATE = 0.35
    LABEL_SMOOTHING = 0.05
    PATIENCE = 15
    WARMUP_EPOCHS = 5


# ==================== HYPERPARAMETER SUGGESTIONS ====================

HYPERPARAMETER_PRESETS = {
    "fast_experiment": {
        "description": "Quick experiment (30 mins)",
        "batch_size": 16,
        "epochs": 30,
        "learning_rate": 1e-4,
        "dropout_rate": 0.2,
    },
    "production": {
        "description": "Target 97-98% accuracy",
        "batch_size": 32,
        "epochs": 100,
        "learning_rate": 5e-5,
        "dropout_rate": 0.35,
    },
    "high_accuracy": {
        "description": "Maximum accuracy (might overfit)",
        "batch_size": 64,
        "epochs": 150,
        "learning_rate": 2e-5,
        "dropout_rate": 0.4,
    },
    "regularized": {
        "description": "Strong regularization (prevent overfitting)",
        "batch_size": 16,
        "epochs": 100,
        "learning_rate": 1e-5,
        "dropout_rate": 0.5,
        "weight_decay": 5e-4,
    },
}


# ==================== AUGMENTATION PRESETS ====================

AUGMENTATION_LEVELS = {
    "light": {
        "rotation": 10,
        "brightness": 0.1,
        "contrast": 0.1,
        "scale": (0.9, 1.1),
    },
    "medium": {  # DEFAULT
        "rotation": 20,
        "brightness": 0.3,
        "contrast": 0.3,
        "scale": (0.85, 1.15),
    },
    "heavy": {
        "rotation": 30,
        "brightness": 0.5,
        "contrast": 0.5,
        "scale": (0.7, 1.3),
    },
}


# ==================== LOSS FUNCTION PRESETS ====================

LOSS_CONFIGS = {
    "balanced": {
        "function": "BCEWithLogitsLoss",
        "pos_weight": None,  # Auto-calculate
        "reduction": "mean",
    },
    "class_weighted": {
        "function": "BCEWithLogitsLoss",
        "pos_weight": 1.0,  # Adjust based on class distribution
        "reduction": "mean",
    },
    "focal": {
        "function": "FocalLoss",
        "alpha": 0.25,
        "gamma": 2.0,
    },
}


# ==================== VALIDATION STRATEGIES ====================

VALIDATION_CONFIGS = {
    "basic": {
        "metric": "accuracy",
        "patience": 10,
        "min_delta": 0.001,  # Minimum improvement
    },
    "conservative": {
        "metric": "val_loss",
        "patience": 20,
        "min_delta": 0.0001,  # Strict improvement threshold
    },
    "aggressive": {
        "metric": "f1_score",
        "patience": 5,
        "min_delta": 0.01,  # Allow larger improvements
    },
}


def get_config(preset="production"):
    """
    Get configuration by preset name
    
    Usage:
        from config import get_config
        config = get_config("production")
        batch_size = config.BATCH_SIZE
    """
    if preset == "tiny":
        return ConfigTiny()
    elif preset == "large":
        return ConfigLarge()
    elif preset == "balanced":
        return ConfigBalanced()
    else:
        return Config()


if __name__ == "__main__":
    # Test configurations
    print("=" * 70)
    print("CONFIGURATION PRESETS")
    print("=" * 70)
    
    for preset_name, preset_config in HYPERPARAMETER_PRESETS.items():
        print(f"\n📋 {preset_name.upper()}")
        print(f"   {preset_config['description']}")
        for key, value in preset_config.items():
            if key != "description":
                print(f"   {key}: {value}")
    
    print("\n" + "=" * 70)
    print("AUGMENTATION LEVELS")
    print("=" * 70)
    
    for level_name, level_config in AUGMENTATION_LEVELS.items():
        print(f"\n📊 {level_name.upper()}")
        for key, value in level_config.items():
            print(f"   {key}: {value}")
    
    print("\n" + "=" * 70)