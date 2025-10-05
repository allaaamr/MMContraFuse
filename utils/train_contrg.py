
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from models.Encoder.genomic import SNN
from models.Fusion.ContRG import ContRGModel

def train_contig(train_loader, val_loader, **kwargs):
    """
    Main training function following ContIG approach
    
    Args:
        train_loader: DataLoader with dict containing 'genomics' and 'radiomics'
        val_loader:  validation DataLoader
        **kwargs: Additional arguments for ContIGModel
    """
    
    # Split kwargs between model and trainer/logger
    allowed_model_keys = {
        'omic_input_dim', 'radio_input_dim', 'radio_type',
        'genomics_output_dim', 'radiomics_output_dim',
        'projection_dim', 'temperature', 'learning_rate',
        'weight_decay', 'max_epochs'
    }
    model_kwargs = {k: v for k, v in kwargs.items() if k in allowed_model_keys}

    # Initialize ContRG model with only supported args
    model = ContRGModel(**model_kwargs)
    
    # Setup trainer
    logger = None
    if kwargs.get('wandb', False):
        logger = WandbLogger(project=kwargs.get('wandb_project', 'MMContraFuse'),
                             name=kwargs.get('wandb_run', None))

    trainer = pl.Trainer(
        max_epochs=kwargs.get('max_epochs', 100),
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        accumulate_grad_batches=kwargs.get('accumulate_grad_batches', 2),
        precision='16-mixed',
        log_every_n_steps=10,
        check_val_every_n_epoch=1 if val_loader else None,
        logger=logger
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    return model