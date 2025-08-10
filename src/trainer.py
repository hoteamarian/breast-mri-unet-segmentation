import torch
from torch.cuda.amp import autocast, GradScaler

class Trainer:
    def __init__(
        self,
        model,
        num_epochs,
        optimizer,
        criterion,
        device,
        scheduler=None,
        scaler: GradScaler = None,
        use_amp: bool = False,
        early_stopping_patience: int = None  # Number of epochs with no improvement before stopping
    ):
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.model = model
        self.device = device
        self.log_interval = 15

        # Mixed precision setup
        self.use_amp = use_amp
        self.scaler = scaler if scaler is not None else (GradScaler() if use_amp else None)

        # Lists to store metric histories
        self.train_losses = []
        self.val_losses = []
        self.train_dices = []
        self.val_dices = []

        # Early stopping parameters
        self.patience = early_stopping_patience
        self.no_improve_epochs = 0
        self.best_dice = 0.0
        self.best_epoch = 0
        self.best_model_state = None

    def dice_coeff(self, logits, target, smooth=1e-6):
        """Compute the Dice coefficient."""
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        intersection = (probs_flat * target_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + target_flat.sum(dim=1)
        dice = (2. * intersection + smooth) / (union + smooth)
        return dice.mean().item()

    def save_checkpoint(self, epoch, dice):
        """Save model checkpoint when validation Dice improves."""
        self.best_dice = dice
        self.best_epoch = epoch
        self.best_model_state = self.model.state_dict()
        filename = f'best_model_epoch{epoch}_dice{dice:.4f}.pth'
        torch.save(self.best_model_state, filename)

    def train(self, train_loader, val_loader):
        """Run training and validation loops with scheduler and early stopping."""
        for epoch in range(1, self.num_epochs + 1):
            train_loss, train_dice = 0.0, 0.0

            # Training phase
            self.model.train()
            for i, (images, masks, labels) in enumerate(train_loader):
                images, masks = images.to(self.device), masks.to(self.device)
                self.optimizer.zero_grad()

                if self.use_amp:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, masks)
                    # backward with scale
                    self.scaler.scale(loss).backward()

                    # unscale gradients before clip
                    self.scaler.unscale_(self.optimizer)
                    # grad clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    # update parameters
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                    loss.backward()

                    # grad clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    self.optimizer.step()

                batch_dice = self.dice_coeff(outputs, masks)
                train_loss += loss.item()
                train_dice += batch_dice

                if (i + 1) % self.log_interval == 0:
                    print(f'Epoch [{epoch}/{self.num_epochs}], Step [{i+1}/{len(train_loader)}], '
                          f'Loss: {loss.item():.4f}, Dice: {batch_dice:.4f}')

            # Validation phase
            val_loss, val_dice = 0.0, 0.0
            self.model.eval()
            with torch.no_grad():
                for images, masks, labels in val_loader:
                    images, masks = images.to(self.device), masks.to(self.device)
                    if self.use_amp:
                        with autocast():
                            outputs = self.model(images)
                            batch_loss = self.criterion(outputs, masks).item()
                    else:
                        outputs = self.model(images)
                        batch_loss = self.criterion(outputs, masks).item()
                    val_loss += batch_loss
                    val_dice += self.dice_coeff(outputs, masks)

            # Calculate epoch averages
            avg_train_loss = train_loss / len(train_loader)
            avg_train_dice = train_dice / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            avg_val_dice = val_dice / len(val_loader)

            print(f'Epoch [{epoch}/{self.num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
            print(f'Epoch [{epoch}/{self.num_epochs}], Train Dice: {avg_train_dice:.4f}, Val Dice: {avg_val_dice:.4f}')

            # Record metrics
            self.train_losses.append(avg_train_loss)
            self.train_dices.append(avg_train_dice)
            self.val_losses.append(avg_val_loss)
            self.val_dices.append(avg_val_dice)

            # Scheduler step if provided
            if self.scheduler is not None:
                self.scheduler.step(avg_val_loss)
                lr = self.optimizer.param_groups[0]['lr']
                print(f'  → new learning rate = {lr:.2e}')

            # Early stopping logic
            if self.patience is not None:
                if avg_val_dice > self.best_dice:
                    self.no_improve_epochs = 0
                    self.save_checkpoint(epoch, avg_val_dice)
                else:
                    self.no_improve_epochs += 1
                    print(f'  → no improvement for {self.no_improve_epochs} epochs (patience={self.patience})')
                if self.no_improve_epochs >= self.patience:
                    print(f'Early stopping at epoch {epoch} after {self.patience} epochs without improvement.')
                    break
            else:
                # Save best model if early stopping is disabled
                if avg_val_dice > self.best_dice:
                    self.save_checkpoint(epoch, avg_val_dice)

    def get_metrics(self):
        """Return training and validation metric histories and best model info."""
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_dices': self.train_dices,
            'val_dices': self.val_dices,
            'best_epoch': self.best_epoch,
            'best_dice': self.best_dice,
        }
