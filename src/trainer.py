import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm


class TMCGTTrainer:
    """Trainer for TMCGT model."""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def train_step(self, batch_data, batch_topo, batch_motif, batch_labels, optimizer):
        """Perform a single training step."""
        self.model.train()
        optimizer.zero_grad()

        batch_data = batch_data.to(self.device)
        batch_topo = batch_topo.to(self.device)
        batch_motif = batch_motif.to(self.device)
        batch_labels = batch_labels.to(self.device)

        logits, graph_repr, total_loss, contrast_loss, topo_loss, motif_loss, weights = self.model(
            batch_data, batch_topo, batch_motif, compute_losses=True
        )

        # Classification loss
        cls_loss = F.cross_entropy(logits, batch_labels)

        # Combined loss
        combined_loss = (cls_loss +
                        self.model.lambda_contrast * contrast_loss +
                        self.model.lambda_topo * topo_loss +
                        self.model.lambda_motif * motif_loss)

        combined_loss.backward()
        optimizer.step()

        return (combined_loss.item(), cls_loss.item(), contrast_loss.item(),
                topo_loss.item(), motif_loss.item(), weights.detach().cpu().numpy())

    def evaluate(self, loader, compute_all_metrics=False):
        """Evaluate the model on a dataset."""
        self.model.eval()
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch_data, batch_topo, batch_motif, batch_labels in loader:
                batch_data = batch_data.to(self.device)
                batch_topo = batch_topo.to(self.device)
                batch_motif = batch_motif.to(self.device)

                logits, _, _, _, _, _ = self.model(batch_data, batch_topo, batch_motif, compute_losses=False)

                probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(batch_labels.cpu().numpy())

        if len(all_labels) == 0:
            return 0.5, 0.5, 0, 0, 0, 0

        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        all_preds = (all_probs > 0.5).astype(int)

        try:
            auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
        except:
            auc = 0.5

        acc = accuracy_score(all_labels, all_preds)

        if compute_all_metrics:
            precision = precision_score(all_labels, all_preds, zero_division=0)
            recall = recall_score(all_labels, all_preds, zero_division=0)
            f1 = f1_score(all_labels, all_preds, zero_division=0)
            return auc, acc, precision, recall, f1

        return auc, acc

    def train_epoch(self, train_loader, optimizer):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for batch_data, batch_topo, batch_motif, batch_labels in train_loader:
            (loss, cls_loss, contrast_loss, topo_loss, motif_loss, weights) = self.train_step(
                batch_data, batch_topo, batch_motif, batch_labels, optimizer
            )

            total_loss += loss

            # Get predictions for accuracy
            batch_data = batch_data.to(self.device)
            batch_topo = batch_topo.to(self.device)
            batch_motif = batch_motif.to(self.device)

            logits, _, _, _, _, _ = self.model(batch_data, batch_topo, batch_motif, compute_losses=False)
            probs = F.softmax(logits, dim=1)[:, 1]
            preds = (probs > 0.5).int().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_labels.cpu().numpy())

        num_batches = len(train_loader)
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        accuracy = accuracy_score(all_labels, all_preds) if all_labels else 0

        return avg_loss, accuracy
