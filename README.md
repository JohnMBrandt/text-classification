## Overview

This is a work-in-progress text classification architecture that jointly learns to classify sentences and documents by multi-task training with an extractive summarizer.

The model is currently a GRU with hierarchical attention using pre-trained gloVe embeddings. Epochs are about 60 seconds on an NVidia Titan X.

#### Model formula

![](https://raw.githubusercontent.com/JohnMBrandt/text-classification/master/model-formula.png)

## Next Steps
Next steps:
  1) add elMo embeddings
  2) Create encoder-decoder network for extractive summarization
  3) Share parameters
  4) Include figure of model structure to readme
  5) Write in ACL format
  
## Installation  
Python 3.6 is required and the dependences can be installed with:

 ```
 python setup.py --install
 ```
## Usage

```
python main.py --epochs {} --training_size {} --batch_size {}
```

