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
        scaler: GradScaler = None,
        use_amp: bool = False
    ):
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.criterion = criterion
        self.model = model
        self.device = device
        self.log_interval = 15

        # Mixed precision flags
        self.use_amp = use_amp
        self.scaler = scaler if scaler is not None else GradScaler() if use_amp else None

        # Lists to store training and validation metrics
        self.train_losses = []
        self.val_losses = []
        self.train_dices = []
        self.val_dices = []

        # Best model and its metrics
        self.best_model = None
        self.best_dice = 0.0
        self.best_epoch = 0

    def dice_coeff(self, logits, target, smooth=1e-6):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        intersection = (probs_flat * target_flat).sum(dim=1)
        unionset = probs_flat.sum(dim=1) + target_flat.sum(dim=1)
        dice = (2. * intersection + smooth) / (unionset + smooth)
        return dice.mean().item()

    def iou(self, pred_mask, true_mask):
        intersection = torch.logical_and(pred_mask, true_mask).sum().item()
        union = torch.logical_or(pred_mask, true_mask).sum().item()
        iou_score = intersection / union if union != 0 else 0.0
        return iou_score

    def save_best_model(self, epoch, dice):
        if dice > self.best_dice:
            self.best_dice = dice
            self.best_epoch = epoch
            self.best_model = self.model.state_dict()

            filename = f'best_model_epoch{epoch}_dice{dice:.4f}.pth'
            torch.save(self.best_model, filename)

    def train(self, train_loader, val_loader):
        for epoch in range(self.num_epochs):
            train_loss = 0.0
            train_dice = 0.0

            # Training loop
            for i, (images, masks, labels) in enumerate(train_loader):
                images, masks = images.to(self.device), masks.to(self.device)

                self.model.train()
                self.optimizer.zero_grad()

                if self.use_amp:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, masks)
                    # scale loss and backpropagate
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                    loss.backward()
                    self.optimizer.step()

                dice = self.dice_coeff(outputs, masks)
                train_loss += loss.item()
                train_dice += dice

                if (i + 1) % self.log_interval == 0:
                    print(
                        f'Epoch [{epoch + 1}/{self.num_epochs}], '
                        f'Step [{i + 1}/{len(train_loader)}], '
                        f'Loss: {loss.item():.4f}, Dice: {dice:.4f}'
                    )

            # Validation loop likewise:
            val_loss = 0.0
            val_dice = 0.0
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

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            avg_train_dice = train_dice / len(train_loader)
            avg_val_dice = val_dice / len(val_loader)

            print(
                f'Epoch [{epoch + 1}/{self.num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}'
            )
            print(
                f'Epoch [{epoch + 1}/{self.num_epochs}], Train Dice: {avg_train_dice:.4f}, Val Dice: {avg_val_dice:.4f}'
            )

            # Save metrics
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(avg_val_loss)
            self.train_dices.append(avg_train_dice)
            self.val_dices.append(avg_val_dice)

            # Save best model
            self.save_best_model(epoch + 1, avg_val_dice)

    def get_metrics(self):
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_dices': self.train_dices,
            'val_dices': self.val_dices,
            'best_model': self.best_model,
            'best_dice': self.best_dice,
            'best_epoch': self.best_epoch
        }
