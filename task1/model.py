import torch
import torch.nn as nn

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, num_convs):
        super(ConvBlock, self).__init__()
        layers = []
        # Use padding='same' so the temporal dimension is preserved
        layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding='same'))
        layers.append(nn.ReLU(inplace=True))
        
        # Add additional convolutional layers if requested
        for _ in range(num_convs - 1):
            layers.append(nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding='same'))
            layers.append(nn.ReLU(inplace=True))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x): 
        return self.block(x)

class MSCNN(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, use_single_stream=False, k_size_stream2=7):
        super(MSCNN, self).__init__()
        self.use_single_stream = use_single_stream
        k1_size = 3 # Stream 1 固定为 3
        
        # --- Stream 1 (k=3) ---
        # Reference design: conv blocks pattern 2-2-3-3-3
        self.s1_conv1 = ConvBlock(in_channels, 64, k1_size, num_convs=2)
        self.s1_pool1 = nn.MaxPool1d(kernel_size=3, stride=3) # temporal downsample x3
        
        self.s1_conv2 = ConvBlock(64, 128, k1_size, num_convs=2)
        self.s1_pool2 = nn.MaxPool1d(kernel_size=3, stride=3) # temporal downsample x3
        
        self.s1_conv3 = ConvBlock(128, 256, k1_size, num_convs=3)
        self.s1_pool3 = nn.MaxPool1d(kernel_size=2, stride=2) # temporal downsample x2
        
        self.s1_conv4 = ConvBlock(256, 512, k1_size, num_convs=3)
        self.s1_pool4 = nn.MaxPool1d(kernel_size=2, stride=2) # temporal downsample x2
        
        self.s1_conv5 = ConvBlock(512, 512, k1_size, num_convs=3)
        self.s1_pool5 = nn.MaxPool1d(kernel_size=2, stride=2) # temporal downsample x2
        
        if not self.use_single_stream:
            # --- Stream 2 (k=k_size_stream2) ---
            # As in the reference design, only the first two blocks use the larger kernel;
            # later blocks reuse the smaller (k=3) kernel to align feature sizes.
            self.s2_conv1 = ConvBlock(in_channels, 64, k_size_stream2, num_convs=2)
            self.s2_pool1 = nn.MaxPool1d(kernel_size=3, stride=3)
            
            self.s2_conv2 = ConvBlock(64, 128, k_size_stream2, num_convs=2)
            self.s2_pool2 = nn.MaxPool1d(kernel_size=3, stride=3)
            
            # Later blocks (3-5) follow the same structure as Stream 1
            self.s2_conv3 = ConvBlock(128, 256, k1_size, num_convs=3)
            self.s2_pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
            
            self.s2_conv4 = ConvBlock(256, 512, k1_size, num_convs=3)
            self.s2_pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
            
            self.s2_conv5 = ConvBlock(512, 512, k1_size, num_convs=3)
            self.s2_pool5 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Adaptive pooling produces [B, C, 1] regardless of input temporal length
        self.global_pool = nn.AdaptiveAvgPool1d(1) 
        
        # --- MLP classifier head ---
        # If using two streams, the concatenated feature dimension doubles
        mlp_in_features = 512 * 2 if not self.use_single_stream else 512
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in_features, 1024), nn.BatchNorm1d(1024), nn.ReLU(inplace=True),
            nn.Linear(1024, 1024), nn.BatchNorm1d(1024), nn.ReLU(inplace=True),
            nn.Linear(1024, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Linear(256, out_channels) # 输出 1 个 logit
        )

    def forward(self, x):
        # Stream 1 forward
        x1 = self.s1_pool1(self.s1_conv1(x))
        x1 = self.s1_pool2(self.s1_conv2(x1))
        x1 = self.s1_pool3(self.s1_conv3(x1))
        x1 = self.s1_pool4(self.s1_conv4(x1))
        x1 = self.s1_pool5(self.s1_conv5(x1))
        x1_pooled = self.global_pool(x1) # [B, 512, 1]
        
        if self.use_single_stream:
            # Flatten [B, 512, 1] -> [B, 512]
            x_flat = x1_pooled.view(x1_pooled.size(0), -1)
        else:
            # Stream 2
            x2 = self.s2_pool1(self.s2_conv1(x))
            x2 = self.s2_pool2(self.s2_conv2(x2))
            x2 = self.s2_pool3(self.s2_conv3(x2))
            x2 = self.s2_pool4(self.s2_conv4(x2))
            x2 = self.s2_pool5(self.s2_conv5(x2))
            x2_pooled = self.global_pool(x2) # [B, 512, 1]
            
            # Concatenate pooled features from both streams
            x_merged = torch.cat((x1_pooled, x2_pooled), dim=1) # [B, 1024, 1]
            # 展平 [B, 1024, 1] -> [B, 1024]
            x_flat = x_merged.view(x_merged.size(0), -1)
        
        # MLP classification head
        output = self.mlp(x_flat) # [B, 1]
        
        return output 