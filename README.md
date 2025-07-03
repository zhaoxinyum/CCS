Convex Combination Star Shape Prior for Data-driven Image Semantic Segmentation
======
## Installation
The code requires python>=3.10, as well as torch>=2.5.1 and torchvision>=0.20.1  
First download a [model checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth),We use 'sam_vit_b_01ec64' as our basic model. Please download this weight and place it in the "parameterfault" folder 

Install relative packages
`pip install -r requirements.txt`
## Data preprocessing
Data preprocessing: Obtain the set of center points  
Sample points on the boundary: 
`python tools/starcenter.py`
Sample points on the skeleton: 
`python tools/opencenter.py`
After preprocessing, the coordinates of the sampled points will be saved, or the vector field will be saved as a float32 pt file
## Training
Optional training methods: lora fine-tuning: --model_name SAM, adding CCS module: --model_name SAMccs, shape loss: --model_name SAMsloss
`python sam_train_with_lora.py --dataname ISIC --model_name SAMccs`
## Evaluation
Please enter the path of the corresponding checkpoint when running
`python eval.py`
