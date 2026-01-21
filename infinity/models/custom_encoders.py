import math
import numpy as np
import torch
import torch.nn as nn
from infinity.utils.eeg_utils import compute_raw_stats, normalize_raw

# Import DE_PSD from the local file DE_PSD.py (assuming it's in the python path or same directory)
# The user specified "from EEG_preprocessing.DE_PSD import DE_PSD", but the file is at ./DE_PSD.py
# We will assume that we need to import it from the current context or provide a wrapper.
# Since we are writing this to infinity/models/custom_encoders.py, we might need to adjust import.
# For now, we will assume DE_PSD is available or copy the function if import is tricky. 
# Given the user context, let's try to import it assuming project root is in path or we copy it.
# To be safe and self-contained, I will include a robust import or a placeholder if it fails,
# but since I read the file content, I can also embed it or place it in infinity/utils if preferred.
# However, the user asked to USE it. Let's try to import from the file I found.

try:
    from DE_PSD import DE_PSD
except ImportError:
    # If standard import fails, try relative or sys path hack, or just define it here to be safe
    # Given I have the content, I will redefine it here or in eeg_utils to avoid import errors 
    # if the file structure is complex. But the user said "@DE_PSD.py ...", let's put it in utils.
    pass

class deepnet(nn.Module):
    def __init__(self, out_dim, C, T):
        super(deepnet, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 25, (1, 10), (1, 1)),
            nn.Conv2d(25, 25, (C, 1), (1, 1)),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),

            nn.Conv2d(25, 50, (1, 10), (1, 1)),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),

            nn.Conv2d(50, 100, (1, 10), (1, 1)),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),

            nn.Conv2d(100, 200, (1, 10), (1, 1)),
            nn.BatchNorm2d(200),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),
        )

        # compute output dimension using a dummy input tensor
        with torch.no_grad():
            dummy = torch.zeros(1, 1, C, T)
            out_features = self.net(dummy).view(1, -1).shape[1]
        self.out_features = out_features
        self.out = nn.Linear(out_features, out_dim)
    
    def forward(self, x, return_features=False):               #input:(batch,1,C,T)
        x = self.net(x)
        x = x.view(x.size(0), -1)
        if return_features:
            return x
        x = self.out(x)
        return x
    
class eegnet(nn.Module):
    def __init__(self, out_dim, C, T):
        super(eegnet, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 8, (1, 64), (1, 1)),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (C, 1), (1, 1)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),
            nn.Conv2d(16, 16, (1, 16), (1, 1)),
            nn.BatchNorm2d(16), 
            nn.ELU(),
            nn.AvgPool2d((1, 2), (1, 2)),
            nn.Dropout2d(0.5)
        )

        # compute output dimension using a dummy input tensor
        with torch.no_grad():
            dummy = torch.zeros(1, 1, C, T)
            out_features = self.net(dummy).view(1, -1).shape[1]
        self.out_features = out_features
        self.out = nn.Linear(out_features, out_dim)
    
    def forward(self, x, return_features=False):               #input:(batch,1,C,T)
        x = self.net(x)
        x = x.view(x.size(0), -1)
        if return_features:
            return x
        x = self.out(x)
        return x

class shallownet(nn.Module):
    def __init__(self, out_dim, C, T):
        super(shallownet, self).__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (C, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 51), (1, 5)),
            #nn.AdaptiveAvgPool2d((1, 26)),
            nn.Dropout(0.5),
        )
        n_samples = math.floor((T - 75) / 5 + 1)
        self.out_features = 40 * n_samples
        self.out = nn.Linear(self.out_features, out_dim)
    
    def forward(self, x, return_features=False):               #input:(batch,1,C,T)
        x = self.net(x)
        x = x.view(x.size(0), -1)
        if return_features:
            return x
        x = self.out(x)
        return x

class mlpnet(nn.Module):
    def __init__(self, out_dim, input_dim):
        super(mlpnet, self).__init__()
        
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, out_dim)
        )
        
    def forward(self, x, return_features=False):               #input:(batch,C,5)
        # Note: mlpnet usually takes features (batch, C*5) or similar flattened
        # The original code had Flatten() as first layer.
        # If return_features is True, we might want intermediate representation?
        # Original code: 512->256->out_dim. 
        # If we act as an encoder, we probably want the 256 dim embedding.
        
        # Manually run through sequential to get intermediate if needed
        # Or just return 256 dim output if return_features=True
        
        x = self.net[0](x) # Flatten
        x = self.net[1](x) # Linear 512
        x = self.net[2](x) # GELU
        x = self.net[3](x) # Linear 256
        x = self.net[4](x) # GELU
        
        if return_features:
            return x # (B, 256)
            
        x = self.net[5](x) # Linear out_dim
        return x

class glmnet(nn.Module):
    """ShallowNet (raw) + MLP (freq) → concat → FC."""

    def __init__(self, occipital_idx, C: int, T: int, *, feat_dim: int = 5, out_dim: int = 40, emb_dim: int = 512):
        super().__init__()
        self.occipital_idx = list(occipital_idx) if occipital_idx is not None else list(range(C))
        self.time_len = T
        self.feat_dim = feat_dim
        
        # Since we are using this as an encoder for alignment (which has its own projector),
        # out_dim here is technically the "embedding dimension" that goes to the alignment projector.
        # However, the original code uses out_dim for classification.
        # For alignment task, we typically ignore the final classification layer if return_features=True.
        # But wait, glmnet combines two branches.
        
        # Global branch processing raw EEG
        self.raw_global = shallownet(emb_dim, C, T)
        # Local branch processing spectral features
        # Input to mlpnet is flattened features: len(occipital_idx) * feat_dim
        self.freq_local = mlpnet(emb_dim, len(self.occipital_idx) * feat_dim)

        # Projection of concatenated features followed by classifier
        # emb_dim from raw_global is (40 * n_samples) -> Wait, shallownet out_dim in original code was passed as first arg?
        # In original code: self.raw_global = shallownet(emb_dim, C, T) -> shallownet(out_dim, C, T)
        # So raw_global outputs `emb_dim` size vector?
        # Let's check shallownet: self.out = nn.Linear(self.out_features, out_dim) -> Yes.
        
        # So:
        # raw_global output: (B, emb_dim)
        # freq_local output: (B, emb_dim) (mlpnet output is out_dim which is emb_dim passed here)
        
        self.projection = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim // 2),
        )
        
        # This is the final output dimension of the encoder system (before alignment projector)
        self.out_features = emb_dim // 2 
        
        self.classifier = nn.Sequential(
            nn.GELU(),
            nn.Linear(emb_dim // 2, out_dim),
        )

    def forward(self, x, return_features: bool = False):
        """
        x: (B, C, T) if passing raw only? 
        Or (B, C, T + feat_dim) if passing combined?
        
        The dataset yields raw EEG (B, 1, C, T) or similar.
        GLMNet requires DE features as well.
        
        If we want to use GLMNet in the alignment pipeline, we need to:
        1. Compute DE features on the fly OR expect input to contain them.
        2. Split input into raw and features.
        
        However, standard dataset currently only provides raw EEG.
        If we strictly follow the request to use GLMNet, we should probably compute features inside forward
        OR modify the dataset to provide them. 
        Computing on the fly on GPU/CPU batch might be slow but easiest for integration.
        
        Let's assume input `x` is just Raw EEG (B, 1, C, T) or (B, C, T).
        We will compute DE features here if possible, or assume x contains them if pre-processed.
        
        Given the previous context, the user wants to replace the code.
        The original `train_classifier_mono.py` computes features using `mlpnet.compute_features` which calls `DE_PSD`.
        
        We should probably integrate feature computation here.
        """
        
        # Handle input shape (B, 1, C, T) -> (B, C, T)
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
            
        B, C, T = x.shape
        
        # Check if features are already appended (hacky check based on time dim)
        # If T == self.time_len, we need to compute features.
        # If T > self.time_len, maybe they are concatenated?
        
        # For this implementation, let's assume we receive raw EEG and compute features on the fly.
        # This requires importing DE_PSD and running it.
        # Note: DE_PSD is CPU based usually (scipy/numpy). Moving back and forth might be slow.
        # But for correctness with "my own encoder applied to EEG signal", this is the way.
        
        x_raw = x
        
        # Compute features
        # We need to run this on CPU numpy
        x_np = x.detach().cpu().numpy()
        
        feats_list = []
        # Import helper inside method to avoid circular imports if any
        from infinity.utils.de_psd import DE_PSD_torch_or_numpy
        
        for i in range(B):
            # x_np[i]: (C, T)
            de = DE_PSD_torch_or_numpy(x_np[i]) 
            feats_list.append(de)
            
        feats = np.array(feats_list) # (B, C, 5)
        x_feat = torch.tensor(feats, device=x.device, dtype=x.dtype)
        
        # Now run the networks
        # shallow net expects (B, 1, C, T)
        # We use return_features=False because in glmnet structure, the sub-modules 
        # are initialized with out_dim=emb_dim, and we want that embedding.
        g_raw = self.raw_global(x_raw.unsqueeze(1), return_features=False) # (B, emb_dim)
        
        # freq_local expects flattened features from occipital channels
        # x_feat: (B, C, 5) -> select occipital -> (B, len(occ), 5) -> flatten -> (B, len(occ)*5)
        feat_occ = x_feat[:, self.occipital_idx, :]
        feat_flat = feat_occ.reshape(B, -1)
        
        l_freq = self.freq_local(feat_flat, return_features=False) # (B, emb_dim)

        features = torch.cat([g_raw, l_freq], dim=1)
        projected = self.projection(features)

        if return_features:
            return projected

        return self.classifier(projected)
