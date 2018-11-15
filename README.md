## Overview

This is a work-in-progress text classification architecture that jointly learns to classify sentences and documents by multi-task training with an extractive summarizer.

The model is currently a GRU with hierarchical attention. 

## Next Steps
Next steps:
  1) add pre-trained embeddings (GloVe and ELmo)
  2) Create encoder-decoder network for extractive summarizatiotn
  3) Share parameters
  
## Installation  
Python 3.6 is required and the dependences can be installed with:

 ```
 python setup.py --install
 ```
## Usage

```
python main.py --epochs {} --training_size {} --batch_size {}
```

