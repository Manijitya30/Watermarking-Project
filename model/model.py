import torch
import torch.nn as nn
import timm
import torch
import torch.nn as nn
import timm


class OptimizedForgeryDetector(nn.Module):
    def __init__(self, dropout_rate=0.4, use_handcrafted=True):
        super().__init__()

        # === BACKBONES ===
        self.convnext = timm.create_model('convnext_base', pretrained=True, num_classes=0)
        self.efficient = timm.create_model('efficientnet_b4', pretrained=True, num_classes=0)

        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=0   # ✅ IMPORTANT FIX
        )

        self.use_handcrafted = use_handcrafted
        self.handcrafted_dim = 39 if use_handcrafted else 0

        # === HANDCRAFTED ===
        if use_handcrafted:
            self.handcrafted_processor = nn.Sequential(
                nn.Linear(self.handcrafted_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),

                nn.Linear(128, 64)
            )
            handcrafted_out = 64
        else:
            handcrafted_out = 0

        # === CNN PROJECTOR ===
        self.cnn_projector = nn.Sequential(
            nn.Linear(1024 + 1792, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256)
        )

        # === VIT PROJECTOR ===
        self.vit_projector = nn.Sequential(
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256)
        )

        # === FINAL FUSION PROJECTOR (🔥 FIX FOR YOUR ERROR) ===
        total_dim = 256 + 256 + handcrafted_out   # 576

        self.fusion_projector = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # === ATTENTION ===
        self.attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=4,
            batch_first=True
        )

        # === CLASSIFIER ===
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, x, handcrafted_features=None):

        # === CNN ===
        f1 = self.convnext(x)
        f2 = self.efficient(x)
        cnn = torch.cat([f1, f2], dim=1)
        cnn = self.cnn_projector(cnn)

        # === VIT ===
        vit = self.vit(x)
        vit = self.vit_projector(vit)

        # === HANDCRAFTED ===
        if self.use_handcrafted and handcrafted_features is not None:
            h = self.handcrafted_processor(handcrafted_features)
            combined = torch.cat([cnn, vit, h], dim=1)
        else:
            combined = torch.cat([cnn, vit], dim=1)

        # === 🔥 FIXED DIMENSION ===
        combined = self.fusion_projector(combined)

        # === ATTENTION ===
        attn_out, _ = self.attention(
            combined.unsqueeze(1),
            combined.unsqueeze(1),
            combined.unsqueeze(1)
        )

        attn_out = attn_out.squeeze(1)

        # === RESIDUAL ===
        fused = combined + attn_out

        # === OUTPUT ===
        return self.classifier(fused)


class LightweightForgeryDetector(nn.Module):
    """
    Lightweight variant for faster training without sacrificing accuracy
    """
    def __init__(self, dropout_rate=0.35, use_handcrafted=True):
        super().__init__()
        
        self.convnext = timm.create_model('convnext_base', pretrained=True, num_classes=0)
        self.efficient = timm.create_model('efficientnet_b4', pretrained=True, num_classes=0)
        
        self.cnn_dim = 1024 + 1792
        self.handcrafted_dim = 39 if use_handcrafted else 0
        self.use_handcrafted = use_handcrafted
        
        if use_handcrafted:
            self.handcrafted_processor = nn.Sequential(
                nn.Linear(self.handcrafted_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout_rate - 0.1),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(dropout_rate - 0.15),
                nn.Linear(64, 32)
            )
            total_dim = self.cnn_dim + 32
        else:
            total_dim = self.cnn_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate - 0.05),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate - 0.1),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate - 0.15),
            
            nn.Linear(128, 1)
        )

    def forward(self, x, handcrafted_features=None):
        f1 = self.convnext(x)
        f2 = self.efficient(x)
        cnn_combined = torch.cat([f1, f2], dim=1)
        
        if self.use_handcrafted and handcrafted_features is not None:
            handcrafted_processed = self.handcrafted_processor(handcrafted_features)
            combined = torch.cat([cnn_combined, handcrafted_processed], dim=1)
        else:
            combined = cnn_combined
        
        out = self.fusion(combined)
        return out