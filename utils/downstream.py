import torch
import torch.nn as nn


# ============================================================================
# Downstream Task Evaluation
# ============================================================================

class DownstreamClassifier(nn.Module):
    """
    Simple classifier for downstream tasks using pre-trained embeddings
    This follows the linear probing protocol in ContIG
    """
    def __init__(self, embedding_dim, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, embeddings):
        return self.classifier(embeddings)


def evaluate_downstream_task(
    contig_model,
    train_loader,
    val_loader,
    num_classes,
    task_type='classification',
    freeze_encoder=True,
    num_epochs=50
):
    """
    Evaluate on downstream tasks using learned embeddings
    
    Two approaches as in ContIG paper:
    1. Linear probing: Freeze encoder, train only classifier
    2. Fine-tuning: Train entire model end-to-end
    
    Args:
        contig_model: Trained ContIG model
        train_loader: Training data (dict with 'genomics', 'radiomics', 'labels')
        val_loader: Validation data
        num_classes: Number of classes for classification
        task_type: 'classification' or 'regression'
        freeze_encoder: If True, only train classifier (linear probing)
        num_epochs: Number of training epochs
    """
    # Freeze encoder if doing linear probing
    if freeze_encoder:
        contig_model.eval()
        for param in contig_model.parameters():
            param.requires_grad = False
    
    # Create downstream classifier
    embedding_dim = contig_model.projection_dim
    downstream_model = DownstreamClassifier(embedding_dim, num_classes)
    
    # Setup for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    contig_model = contig_model.to(device)
    downstream_model = downstream_model.to(device)
    
    optimizer = torch.optim.Adam(downstream_model.parameters(), lr=1e-3)
    

    criterion = nn.CrossEntropyLoss()

    
    # Training loop
    best_val_acc = 0
    for epoch in range(num_epochs):
        downstream_model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            genomics = batch['genomics'].to(device)
            radiomics = batch['radiomics'].to(device)
            labels = batch['labels'].to(device)
            
            # Get embeddings from ContIG model
            embeddings = contig_model.get_embeddings(genomics, radiomics)
            
            # Forward through classifier
            outputs = downstream_model(embeddings)
            loss = criterion(outputs, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if task_type == 'classification':
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        # Validation
        if val_loader:
            val_acc = validate_downstream(
                contig_model, downstream_model, val_loader, 
                device, task_type
            )
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print(f'Epoch {epoch+1}: New best validation accuracy: {val_acc:.4f}')
        
        if task_type == 'classification':
            train_acc = 100. * correct / total
            print(f'Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, '
                  f'Train Acc: {train_acc:.2f}%')
    
    return downstream_model, best_val_acc


def validate_downstream(contig_model, classifier, val_loader, device, task_type):
    """Validate downstream task performance"""
    contig_model.eval()
    classifier.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            genomics = batch['genomics'].to(device)
            radiomics = batch['radiomics'].to(device)
            labels = batch['labels'].to(device)
            
            embeddings = contig_model.get_embeddings(genomics, radiomics)
            outputs = classifier(embeddings)
            
            if task_type == 'classification':
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
    
    return 100. * correct / total if task_type == 'classification' else 0
