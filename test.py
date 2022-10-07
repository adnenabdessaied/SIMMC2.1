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
import os.path


def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=os.path.join(os.path.dirname(__file__),  'checkpoints/OLViT_phase_1.ckpt'),
        help="The path to the trained model."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=os.path.join(os.path.dirname(__file__),  'config/OLViT_phase_1_test.json'),
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

    if 'output_path' not in config['checkpoint'].keys():
        raise Exception('no output path provided in config (full path for disc model only path to output folder for gen. model)')

    

    if config['training']['seed'] != None:
        pl.seed_everything(config['training']['seed'])

    trainer = Trainer(
        accelerator='gpu',
        devices=[args.gpu]
    )
    data = Simmc2Data(config=config)

    model = GenerativeModel(config=config, output_path=config['checkpoint']['output_path'])
    trainer.test(model=model, ckpt_path=args.checkpoint_path, dataloaders=data)


if __name__ == '__main__':
    args = process_args()
    main(args)