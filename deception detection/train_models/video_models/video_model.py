

def cleanup_memory():
    """Clears GPU cache and RAM to free up memory."""
    torch.cuda.empty_cache()  # Clears the GPU memory cache
    gc.collect()


# ========================
# Video Model Definition
# ========================
class VideoModel(nn.Module):
    def __init__(self, base_model_type, num_encoders, adapter, adapter_type):
        super(VideoModel, self).__init__()
        self.num_encoders = num_encoders
        self.adapter = adapter
        self.adapter_type = adapter_type

        # Define base model based on type
        if base_model_type == 'vit_custom':
            self.base_model = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1),
                nn.AdaptiveAvgPool2d(1)
            )
            self.feature_dim = 256
        elif base_model_type == 'resnet50':
            resnet = torchvision.models.resnet50(pretrained=True)
            for p in resnet.parameters():
                p.requires_grad = False
            self.base_model = nn.Sequential(*list(resnet.children())[:-2])
            self.feature_dim = 2048
        elif base_model_type == 'efficientnet_b0':
            efficientnet = torchvision.models.efficientnet_b0(pretrained=True)
            for p in efficientnet.parameters():
                p.requires_grad = False
            self.base_model = efficientnet.features
            self.feature_dim = 1280

        # Feature projection
        self.feature_projection = nn.Linear(self.feature_dim, 768)

        # ViT encoder setup
        vit = torchvision.models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        for p in vit.parameters():
            p.requires_grad = False
        self.encoder = nn.Sequential(*[
            self._make_adapter_layer(vit.encoder.layers[i])
            for i in range(num_encoders)
        ])

        # Classification head
        self.classifier = nn.Linear(768, 2)

    def _make_adapter_layer(self, original_layer):
        if self.adapter:
            return nn.Sequential(
                original_layer,
                nn.Linear(768, 768),
                nn.ReLU()
            )
        return original_layer

    def forward(self, x):
        batch_size, num_frames = x.shape[0], x.shape[1]
        x = x.view(-1, *x.shape[2:])
        features = self.base_model(x)
        features = nn.functional.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
        features = features.view(batch_size, num_frames, -1)
        features = self.feature_projection(features)
        features = self.encoder(features)
        return self.classifier(features.mean(1))


# ========================
# Dataset Class
# ========================
class VisualDataset(Dataset):
    def __init__(self, annotations_file, frames_dir, num_frames=64, img_size=160, is_train=True):
        self.annos = pd.read_csv(annotations_file)
        self.frames_dir = frames_dir
        self.num_frames = num_frames
        self.img_size = img_size
        self.is_train = is_train

        # Filter out missing clips
        self.valid_data = []
        for idx in range(len(self.annos)):
            clip_name = self.annos.iloc[idx, 0]
            clip_dir = os.path.join(self.frames_dir, clip_name)
            if self._validate_clip(clip_dir):
                label = self._process_label(self.annos.iloc[idx, 1])
                self.valid_data.append((clip_dir, label))
            else:
                print(f"Missing frames for clip: {clip_name}")

        print(f"Found {len(self.valid_data)}/{len(self.annos)} valid clips.")

        # Define transforms
        if self.is_train:
            self.transforms = T.Compose([
                T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transforms = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def _validate_clip(self, clip_dir):
        if not os.path.exists(clip_dir):
            return False
        frame_files = glob.glob(os.path.join(clip_dir, "*.jpg"))
        return len(frame_files) >= self.num_frames

    def _process_label(self, label_str):
        str_label = str(label_str).strip().lower()
        return 0 if any(k in str_label for k in ['truth', '0']) else 1

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        clip_dir, label = self.valid_data[idx]
        try:
            frame_files = sorted(glob.glob(os.path.join(clip_dir, "*.jpg")))
            frame_indices = torch.linspace(0, len(frame_files)-1, self.num_frames).long()
            frames = [self.transforms(Image.open(frame_files[i]).convert('RGB')) for i in frame_indices]
            return torch.stack(frames), torch.tensor(label)
        except Exception as e:
            print(f"Error loading {clip_dir}: {e}")
            return None, None


def visual_collate_fn(batch):
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        return torch.zeros(1, 64, 3, 160, 160), torch.tensor([0])
    frames, labels = zip(*batch)
    return torch.stack(frames), torch.stack(labels)


# ========================
# Training Function
# ========================
def train_visual_model():
    # Base configuration
    base_config = {
        'num_epochs': 20,
        'protocols_dir': r"/content/drive/MyDrive/google_collab/Training_Protocols",
        'frames_dir': r'/content/face_frames/face_frames',
        'checkpoint_dir': r'/content/drive/MyDrive/Training_Protocols/video_models',
        'log_file': r'/content/drive/MyDrive/Training_Protocols/video_models/training_log_male_female.csv'
    }
    

    # Model configurations to try
    model_configs = [
        {'base_model_type': 'resnet50', 'num_encoders': 2, 'adapter': True, 'adapter_type': 'nlp', 'lr': 5e-5, 'batch_size': 16},
    ]

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(base_config['checkpoint_dir'], exist_ok=True)

    # Initialize log file
    log_file = base_config['log_file']
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write('config_id,epoch,train_loss,val_loss,val_accuracy,val_f1,val_auc\n')

    # Define training and validation files
    train_file = f"{base_config['protocols_dir']}/male.csv"
    val_file = f"{base_config['protocols_dir']}/female.csv"

    # Loop through model configurations
    for config_id, model_config in enumerate(model_configs):
        print(f"\nTraining model config {config_id}: {model_config}")

        # Create datasets
        train_ds = VisualDataset(train_file, base_config['frames_dir'], is_train=True)
        val_ds = VisualDataset(val_file, base_config['frames_dir'], is_train=False)

        # Create data loaders
        train_loader = DataLoader(train_ds, batch_size=model_config['batch_size'], collate_fn=visual_collate_fn, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=model_config['batch_size'], collate_fn=visual_collate_fn, shuffle=False)

        # Initialize model
        model = VideoModel(
            base_model_type=model_config['base_model_type'],
            num_encoders=model_config['num_encoders'],
            adapter=model_config['adapter'],
            adapter_type=model_config['adapter_type']
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=model_config['lr'])
        criterion = nn.CrossEntropyLoss()
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
        scaler = GradScaler()

        # Training loop
        for epoch in range(base_config['num_epochs']):
            model.train()
            total_train_loss = 0
            for frames, labels in train_loader:
                frames, labels = frames.to(device), labels.to(device)
                optimizer.zero_grad()
                with autocast():
                    outputs = model(frames)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_train_loss += loss.item()
            avg_train_loss = total_train_loss / len(train_loader)

            cleanup_memory()  # Cleanup memory

            # Validation
            model.eval()
            val_loss = 0
            all_labels, all_preds, all_probs = [], [], []
            with torch.no_grad():
                for frames, labels in val_loader:
                    frames, labels = frames.to(device), labels.to(device)
                    outputs = model(frames)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    probs = torch.softmax(outputs, dim=1)[:, 1]
                    preds = torch.argmax(outputs, dim=1)
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = accuracy_score(all_labels, all_preds)
            val_f1 = f1_score(all_labels, all_preds, average='binary')
            val_auc = roc_auc_score(all_labels, all_probs)

            # Log metrics
            with open(log_file, 'a') as f:
                f.write(f"{config_id},{epoch+1},{avg_train_loss:.4f},{avg_val_loss:.4f},{val_accuracy:.4f},{val_f1:.4f},{val_auc:.4f}\n")
            print(f"Config {config_id} | Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Acc: {val_accuracy:.4f} | F1: {val_f1:.4f} | AUC: {val_auc:.4f}")

            scheduler.step(avg_val_loss)

        cleanup_memory()  # Final cleanup

    print("All training complete!")


if __name__ == "__main__":
    train_visual_model()

this is my code , i want to draw the model archietecture of resent50 and efficientnet_bo model 