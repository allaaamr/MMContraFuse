
import torch
import pytorch_lightning as pl
from models.Encoder.genomic import SNN
from models.Fusion import ContRG

def train_contig(train_loader, val_loader=None, **kwargs):
    """
    Main training function following ContIG approach
    
    Args:
        train_loader: DataLoader with dict containing 'genomics' and 'radiomics'
        val_loader: Optional validation DataLoader
        **kwargs: Additional arguments for ContIGModel
    """
    
    # Initialize ContRG model
    model = ContRG(
        **kwargs
    )
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=kwargs.get('max_epochs', 100),
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        accumulate_grad_batches=kwargs.get('accumulate_grad_batches', 2),
        log_every_n_steps=10,
        check_val_every_n_epoch=1 if val_loader else None
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    return model