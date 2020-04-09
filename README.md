# LSC-CNN

This repository is the pytorch implementation for the crowd counting model, LSC-CNN, proposed in the paper - [**Locate, Size and Count: Accurately Resolving People in Dense Crowds via Detection**](https://arxiv.org/pdf/1906.07538.pdf).

If you find this work useful in your research, please consider citing the paper:
```
@article{LSCCNN20,
    Author = {Sam, Deepak Babu and Peri, Skand Vishwanath and Narayanan Sundararaman, Mukuntha,  and Kamath, Amogh and Babu, R. Venkatesh},
    Title = {Locate, Size and Count: Accurately Resolving People in Dense Crowds via Detection},
    Journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
    Year = {2020}
}
```
# Requirements
We strongly recommend to run the codes in NVidia-Docker. Install both `docker` and `nvidia-docker` (please find instructions from their respective installation pages).
After the docker installations, pull pytorch docker image with the following command:
`docker pull nvcr.io/nvidia/pytorch:18.04-py3`
and run the image using the command:
`nvidia-docker run --rm -ti --ipc=host nvcr.io/nvidia/pytorch:18.04-py3`

Further software requirements are listed in `requirements.txt`. 

To install them type, `pip install -r requirements.txt`

The code has been run and tested on `Python 3.6.3`, `Ubuntu 14.04.5 LTS` and `Cuda 9.0, V9.0.176`. 

_Please NOTE that `Python 2.7` is not supported and the code would ONLY work on `Python 3` versions._

# Dataset Download
Download Shanghaitech dataset from [here](https://github.com/desenzhou/ShanghaiTechDataset).
Download UCF-QNRF dataset from [here](http://crcv.ucf.edu/data/ucf-qnrf/).

Place the dataset in `../dataset/` folder. (`dataset` and `lsc-cnn` folders should have the same parent directory). So the directory structure should look like the following:
```
-- lsc-cnn
   -- network.py
   -- main.py
   -- ....
-- dataset
   --STpart_A
     -- test_data
	    -- ground-truth
	    -- images
     -- train_data
	    -- ground-truth
	    -- images
  --UCF-QNRF
    --train_data
      -- ...
    --test_data
      -- ...
```

## Pretrained Models
The pretrained models for testing can be downloaded from [here](https://drive.google.com/open?id=1hlJg4ux_BI3z_8zRdwwE7oQoumzSYIEg).

For evaluating on any pretrained model, place the corresponding `models` from the aforementioned link to `lsc-cnn` folder and follow instructions in Testing section.

# Usage
 Clone the repository.
`git clone https://github.com/val-iisc/lsc-cnn.git`

`cd lsc-cnn`

`pip install -r requirements.txt`

Download `models` folders to `lsc-cnn`.

Download Imagenet pretrained VGG weights from [here](https://drive.google.com/open?id=1hlJg4ux_BI3z_8zRdwwE7oQoumzSYIEg) (Download the `imagenet_vgg_weights` folder) and place it in the parent directory of `lsc-cnn`.

## Preparing the Dataset
Run the following code to dump the dataset for `lsc-cnn`

`python main.py --dataset="parta" --gpu=<gpu_number>`

*Warning : If the dataset is already prepared, this command would start the training!*

*Dataset dump size for `ST_PartA is ~13 GB`, for `QNRF is ~150 GB`, and for `ST_PartB is ~35 GB`, so make sure there is sufficient disk space before training/testing.*

## Training
- For training `lsc-cnn` run:

`python main.py --dataset="parta" --gpu=2 --start-epoch=0 --epochs=30`

```
--dataset = parta / ucfqnrf / partb
--gpu = GPU number
--epochs = Number of epochs to train. [For QNRF set --epochs=50]
--patches = Number of patches to crop per image [For QNRF use --patches=30, for other crowd counting dataset default parameter --patches=100 works.]
```

## Testing
### For testing on Part-A

`python main.py --dataset="parta" --gpu=2 --start-epoch=13 --epochs=13 --threshold=0.21`

### For testing on Part-B

`python main.py --dataset="partb" --gpu=2 --start-epoch=24 --epochs=24 --threshold=0.25`

### For testing on QNRF

`python main.py --dataset="ucfqnrf" --gpu=2 --start-epoch=46 --epochs=46 --threshold=0.20`

- All the metrics are displayed once the above code completes its run.

- To do a threshold test, just remove the `--threshold` flag:

For example:


`python main.py --dataset="parta" --gpu=2 --start-epoch=13 --epochs=13`

Use the `--mle` option to compute the mean localization error. If using MLE, compile the function first:
```
cd utils/mle_function
./script.sh
```
This generates an `error_function.so` file in the `./lsc-cnn` directory which is used by `main.py` for computing the MLE metric.

# Test Outputs
Test outputs consist of box predictions for validation set at `models/dump` and that of the test set at `models/dump_test`.

## Contact
For further queries, please mail at `pvskand <at> protonmail <dot> com`.

