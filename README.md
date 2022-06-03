# co-mod-gan-pytorch
Implementation of the paper ``Large Scale Image Completion via Co-Modulated Generative Adversarial Networks"

official tensorflow version: https://github.com/zsyzzsoft/co-mod-gan

Input image<img src="imgs/ffhq_in.png" width=200> Mask<img src="imgs/ffhq_m.png" width=200>  Result<img src="imgs/example_output.jpg" width=200>  

## Requirements

- Linux Machine
- NVIDIA GPU

## Setup

If you are running this script on a remote machine, you can create a Jupyter Notebook config file named `jupter_notebook_config.py`.

In my case, it contains :
``
c.NotebookApp.custom_display_url = 'http://some_hostname:8888'
``

## Usage

```
source setup_env.sh
python -m jupyter notebook --no-browser --port=8888 --ip=0.0.0.0 notebooks
```

converted model:
* FFHQ 512 checkpoints/co-mod-gan-ffhq-9-025000.pth
* FFHQ 1024 checkpoints/co-mod-gan-ffhq-10-025000.pth
* Places 512 checkpoints/co-mod-gan-places2-050000.pth


## Reference

[1] official tensorflow version: https://github.com/zsyzzsoft/co-mod-gan

[2] stylegan2-pytorch https://github.com/rosinality/stylegan2-pytorch

[3] Co-mod-gan PyTorch fork https://github.com/styler00dollar/Colab-co-mod-gan-pytorch