# Documentation

## Repository Structure

- config - contains all configs for all experiments as well as the defualt config
- model_analysis - contains the model output (json for gen, pkl for disc) and some analysis notebooks
- src - contains the actual code
    - combiner: all 4 combiner options introduced in the thesis
    - data_modules: pytorch lightning data modules for the different datasets
    - models: contains the code for the the generative and the discriminative model and the logic to update and use the states 
    - state_trackers: different versions for the actual state tracker module
    - utils: dataloaders, helper functions and code which was taken from other repositories

## Model 

<p align="center">
<img src="img/inheritance_structure.png" width="50%" />

## Config
### Sections
- wandb: all parameters for logging with weights and biases (experiment name, tags, ...)
- model: 
    - model_type: which model to use (discriminative, generative)
    - dataset: (dvd, avsd, simmc2)
    - feature_type: relevant for simmc2 (objext_text_features, resnet50)
    - ... 
    - more hyperparams
    - ...
    - projection_as_in_aloe: if true, emb_dim of transformer is n_heads * v_emb_dim (as done in the Aloe paper)
- extended_model: hyperparameters related to the state tracker
- training: training hyperparameters
- datamodule: 
    - fea_dir: path where the visual features are stored 
    - data_dir: path where the actual dialogues are stored
- checkpoint: where to save the data or 

### Working Config Examples
- generative: /scratch/hochmeister/code/msc2022_hochmeister/config/experiments/simmc2_ablation_object_text/obj_text_feat_both_state_vectors.json
- discriminative: /scratch/hochmeister/code/msc2022_hochmeister/config/experiments/exp15_ablation_state_vectors/both_state_vectors.json

## How to train
- write a new config or adapt an existing one from the config folder
    - set model type, dataset, feature type and so on 
- change the path in the train.py file
- change up the necessary parameters for the pytorch lightning trainer in the train.py file (which gpu to use, how often to validate, ...) 
- execute train.py

## How to test
- choose the checkpoint path from the checkpoint folder
- set the chkpt_pth variable in test.py
- change the config path to a fitting config for the checkpoint 
- set the output_path in the config file (full path incl. filename for disc. model, only folder path for gen. model)
- execute test.py

## Other repositories and code snippets used for this thesis
### Repositories
- https://github.com/facebookresearch/DVDialogues
- https://github.com/salesforce/VD-BERT
- https://github.com/batra-mlp-lab/avsd
- https://github.com/facebookresearch/simmc2


### Snippets
- https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer
- https://github.com/pytorch/pytorch/issues/68407
- https://gist.github.com/airalcorn2/50ec06517ce96ecc143503e21fa6cb91
- https://pytorch.org/tutorials/beginner/translation_transformer.html





