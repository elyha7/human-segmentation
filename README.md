# Human Segmentation

## Description
Deep person segmentation and visual transformation of your photos. Currently supported visual transforms:
* Bokeh effect
* Black and white background
* Layer of mask above person
## Examples
Bokeh effect

<img src="/test_photo/transformed/bokeh2.jpg" width="400"/>

Black and white background

<img src="/test_photo/transformed/bnw.jpg" width="400"/>

Layered mask effect

<img src="/test_photo/transformed/layered.jpg" width="400"/>

## Installation

* Install requirements: `pip3 install -r requirements.txt`
* Download resnet-50 model weights (about 150mb) from [google drive](https://drive.google.com/drive/folders/1JbiE0WM-3vCw2VttMr3fBZBVHiD2gLMu?usp=sharing)
* Put weights in `weights/` folder
## Training
Model was trained with [supervisely person segmentation dataset](https://supervise.ly/explore/projects/supervisely-person-dataset-23304/datasets). In `utils/` folder you can find scripts, that convert supervisely format annotations into binary masks.

You can train any model from qubvel repo. I used Unet architecture with resnet-50 backbone as main model and more lightweight efficientnet-b1.

Notebooks with training code can be found in `notebooks/` folder.
## Usage
Results available via command line interface. There are some keys:
* `--model_path` - path to model weights
* `--device` - device to run inference on: gpu or cpu
* `--trans_type` - type of visual transformation effect: bokeh, bnw, layered.
* `--result_path` - path to save resulting transformed image.
* `--blur_power` - only for bokeh transformation option, strength of gaussian blur. Int value from 1 to 3, 1 - the weakest, 3 - the strongest.
#### Example of console script:
```bash
python3 inference.py 'test_photo/medium.jpg' --model_path 'weights/resnet50_089.pb' --device 'gpu' --trans_type 'layered' --result_path 'test1.jpg' --blur_power 2

```
## Citiation
Thanks [qubvel](https://github.com/qubvel/segmentation_models.pytorch) for pytorch image segmentation library.
