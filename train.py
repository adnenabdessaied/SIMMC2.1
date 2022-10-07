from src.models.generative_model import GenerativeModel
from src.data_modules.simmc2_data import Simmc2Data
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
import wandb
from config.config import read_default_config, read_config, update_nested_dicts
import flatdict
import argparse
import os


def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default=os.path.join(os.path.dirname(__file__),  'config/OLViT_phase_1_train.json'),
        help="The path to the test config file."
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="Gpu id."
    )  
    parser.add_argument(
        "--image_feature_path",
        type=str,
        #required=True,
        help="Please provide the path to the resnet50 image features",
        default="/scratch/hochmeister/simmc2/data/visual_features_resnet50_simmc2.1.pt"
    ) 
    args = parser.parse_args()
    return args


def update_config_paths(config):
    for section in ['datamodule', 'checkpoint']:
        for key, value in config[section].items():
            config[section][key] = os.path.join(os.path.dirname(__file__),  value)
    return config


def main(args):

    # read the default conifg and update the values with the experiment specific config
    config = read_default_config()
    experiment_config = read_config(args.config_path)
    config = update_nested_dicts(old_dict=config, update_dict=experiment_config)
    config = update_config_paths(config)
    config['datamodule']['fea_dir'] = args.image_feature_path


    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        monitor='bleu4', mode="max",      
        save_top_k=1,           # Save only the model with the best dev accuracy    
        dirpath=config["checkpoint"]["checkpoint_folder"],     # "." by default set in config.py
        filename=config["checkpoint"]["checkpoint_file_name"],    # Use config.checkpoint_save_name, config.name (WandB run name) or None in that order
        every_n_epochs=1    # Check every epoch and replace checkpoint if score is higher than last best  
    )
    lr_monitor_cb = LearningRateMonitor(
        logging_interval='step'
    )

    callbacks = []
    callbacks.append(checkpoint_cb)
    callbacks.append(lr_monitor_cb)
    

    if config['training']['seed'] != None:
        pl.seed_everything(config['training']['seed'])

    trainer = Trainer(
        accelerator='gpu',
        devices=args.gpu,
        fast_dev_run=False,
        max_epochs=config['training']['epochs'],
        check_val_every_n_epoch=1,
        log_every_n_steps=5,
        strategy=pl.strategies.ddp.DDPStrategy(find_unused_parameters=False),
        accumulate_grad_batches=config['training']['accumulate_grad_batches'],
        precision=16,
        callbacks=callbacks
    )
    data = Simmc2Data(config=config)
    model = GenerativeModel(config=config)

    trainer.fit(model, data)

if __name__ == '__main__':
    args = process_args()
    main(args)