

    for fold_file in train_files:
        annos = pd.read_csv(fold_file)
        for idx in range(len(annos)):
            total_files += 1
            clip_name = annos.iloc[idx, 0]
            audio_path = os.path.join(audio_dir, f"{clip_name}.wav")
            if os.path.exists(audio_path):
                try:
                    metadata = torchaudio.info(audio_path)
                    duration = metadata.num_frames / metadata.sample_rate
                    durations.append(duration)
                    sample_rates.add(metadata.sample_rate)
                    label = 0 if any(k in str(annos.iloc[idx, 1]).lower() for k in ['truth', '0']) else 1
                    class_counts[label] += 1
                    valid_files += 1
                except Exception as e:
                    print(f"Error analyzing {audio_path}: {e}")

    duration_stats = {
        'min': min(durations),
        'max': max(durations),
        'mean': np.mean(durations),
        'median': np.median(durations),
        'std': np.std(durations)
    }
    return {'class_counts': class_counts, 'duration_stats': duration_stats, 'sample_rates': sample_rates}

# Audio Dataset Class
class AudioDataset(Dataset):
    def __init__(self, annotations_file, audio_dir, target_length=52000, augmentations=False, use_spectrogram=False):
        self.annos = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.target_length = target_length
        self.augmentations = augmentations
        self.use_spectrogram = use_spectrogram
        self.spec_transform = T.MelSpectrogram(sample_rate=16000, n_mels=128, n_fft=1024, hop_length=512)
        self.valid_data = [(os.path.join(audio_dir, f"{self.annos.iloc[idx, 0]}.wav"),
                           0 if any(k in str(self.annos.iloc[idx, 1]).lower() for k in ['truth', '0']) else 1)
                          for idx in range(len(self.annos)) if os.path.exists(os.path.join(audio_dir, f"{self.annos.iloc[idx, 0]}.wav"))]

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        audio_path, label = self.valid_data[idx]
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0)
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

        if waveform.shape[0] > self.target_length:
            start = torch.randint(0, waveform.shape[0] - self.target_length, (1,))
            waveform = waveform[start:start+self.target_length]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, self.target_length - waveform.shape[0]))

        if self.augmentations and torch.rand(1) < 0.5:
            waveform = self._apply_augmentations(waveform)

        if self.use_spectrogram:
            spectrogram = self.spec_transform(waveform)
            spectrogram = T.AmplitudeToDB()(spectrogram).unsqueeze(0)
            return spectrogram, torch.tensor(label)
        return waveform, torch.tensor(label)

    def _apply_augmentations(self, waveform):
        if torch.rand(1) < 0.3:
            noise = torch.randn(waveform.shape) * 0.01
            waveform = waveform + noise
        return waveform

# Collate Function
def audio_collate_fn(batch):
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        return torch.zeros(1, 16000), torch.tensor([0])
    inputs, labels = zip(*batch)
    inputs = torch.stack(inputs)
    labels = torch.stack(labels)
    return inputs, labels

# Compute Class Weights
def compute_class_weights(protocols_dir, audio_dir):
    train_files = glob.glob(f"{protocols_dir}/train_fold*.csv")
    labels = []
    for fold_file in train_files:
        annos = pd.read_csv(fold_file)
        for idx in range(len(annos)):
            clip_name = annos.iloc[idx, 0]
            audio_path = os.path.join(audio_dir, f"{clip_name}.wav")
            if os.path.exists(audio_path):
                label = 0 if any(k in str(annos.iloc[idx, 1]).lower() for k in ['truth', '0']) else 1
                labels.append(label)
    class_counts = np.bincount(labels)
    weights = len(labels) / (2.0 * class_counts)
    return torch.tensor(weights, dtype=torch.float)

# Compute Metrics
def compute_metrics(preds, labels):
    preds = preds.argmax(dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    auc = roc_auc_score(labels, preds) if len(np.unique(labels)) > 1 else 0
    return acc, f1, recall, auc

# Model Definitions (unchanged from previous code for brevity)
class Wav2Vec2Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base", num_labels=2)
    def forward(self, x):
        return self.model(x).logits

# Training Function
def train_model(model_class, model_params, hyperparams, config, fold):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_class(**model_params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
    criterion = nn.CrossEntropyLoss(weight=compute_class_weights(PROTOCOLS_DIR, AUDIO_DIR).to(device))

    train_dataset = AudioDataset(f"{PROTOCOLS_DIR}/train_fold{fold}.csv", AUDIO_DIR, config['target_length'],
                                 augmentations=True, use_spectrogram=config['use_spectrogram'])
    val_dataset = AudioDataset(f"{PROTOCOLS_DIR}/test_fold{fold}.csv", AUDIO_DIR, config['target_length'],
                               use_spectrogram=config['use_spectrogram'])

    train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True, collate_fn=audio_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=hyperparams['batch_size'], collate_fn=audio_collate_fn)

    best_val_loss = float('inf')
    patience_counter = 0
    model_checkpoint_dir = os.path.join(CHECKPOINT_DIR, f"{model_class.__name__}_fold{fold}")
    os.makedirs(model_checkpoint_dir, exist_ok=True)

    for epoch in range(hyperparams['num_epochs']):
        model.train()
        train_loss = 0
        train_preds, train_labels = [], []
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_preds.append(outputs.detach())
            train_labels.append(labels)

        model.eval()
        val_loss = 0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                val_preds.append(outputs)
                val_labels.append(labels)

        train_preds = torch.cat(train_preds)
        train_labels = torch.cat(train_labels)
        train_acc, train_f1, train_recall, train_auc = compute_metrics(train_preds, train_labels)
        val_preds = torch.cat(val_preds)
        val_labels = torch.cat(val_labels)
        val_acc, val_f1, val_recall, val_auc = compute_metrics(val_preds, val_labels)

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        metrics = {
            'model': model_class.__name__,
            'hyperparams': str(hyperparams),
            'fold': fold,
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_f1': train_f1,
            'train_recall': train_recall,
            'train_auc': train_auc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'val_recall': val_recall,
            'val_auc': val_auc
        }
        pd.DataFrame([metrics]).to_csv(METRICS_FILE, mode='a', header=not os.path.exists(METRICS_FILE), index=False)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint_path = os.path.join(model_checkpoint_dir, f"best_model_lr{hyperparams['learning_rate']}_bs{hyperparams['batch_size']}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }, checkpoint_path)
            print(f"Fold {fold}, Epoch {epoch + 1}: Saved best model with val_loss {val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"Early stopping at epoch {epoch + 1} for fold {fold}")
                break

# Main Execution
if __name__ == "__main__":
    # Analyze dataset to set target length
    dataset_stats = analyze_dataset(PROTOCOLS_DIR, AUDIO_DIR)
    TARGET_LENGTH = 52000

    # Base configuration
    config = {
        'target_length': TARGET_LENGTH,
        'use_spectrogram': True,
        'patience': 5
    }

    # Model and Hyperparameter Configurations
    models = {
        'Wav2Vec2Model': {'class': Wav2Vec2Model, 'params': {}, 'use_spectrogram': False,
                         'grid': [{'learning_rate': 1e-5, 'batch_size': 16, 'num_epochs': 20},
                            }

    # Ensure metrics file and checkpoint directory exist
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    if not os.path.exists(METRICS_FILE):
        pd.DataFrame(columns=['model', 'hyperparams', 'fold', 'epoch', 'train_loss', 'train_acc', 'train_f1',
                              'train_recall', 'train_auc', 'val_loss', 'val_acc', 'val_f1',
                              'val_recall', 'val_auc']).to_csv(METRICS_FILE, index=False)

    # Get available folds
    train_files = glob.glob(f"{PROTOCOLS_DIR}/train_fold*.csv")
    fold_numbers = sorted([f.split("train_fold")[1].split(".csv")[0] for f in train_files])

    # Run each model with its hyperparameter configurations across all folds
    for model_name, info in models.items():
        print(f"\nTraining {model_name}")
        config['use_spectrogram'] = info.get('use_spectrogram', True)
        for hyperparams in info['grid']:
            print(f"Hyperparams: {hyperparams}")
            for fold in fold_numbers:
                print(f"Processing Fold {fold}")
                train_model(info['class'], info['params'], hyperparams, config, fold)
