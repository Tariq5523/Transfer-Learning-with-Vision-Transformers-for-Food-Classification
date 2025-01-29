# Food Classification Using Vision Transformers
This project investigates the application of Vision Transformers, specifically the TransNeXt architecture, for food image classification using the Food101 dataset. Four variants of the TransNeXt model—Micro, Tiny, Small, and Base—were fine-tuned after being pre-trained on ImageNet-1K. Advanced augmentation techniques like MixUp and CutMix were used to improve generalization. Experimental results showed that the TransNeXt-Small model achieved the highest top-1 accuracy of 91.26%, surpassing even larger models with significantly fewer parameters.
# How to Run
1. Download the [Food101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) dataset.
2. Install Requirements using 'pip install -r requirements.txt'
3. Download Pre-Trained Models --> [TransNeXt](https://github.com/DaiShiResearch/TransNeXt/tree/main/classification)
4. Ensure the appropriate model is selected in dist_train.bat
5. Start the training process by running: 'dist_train.bat'
6. For visualizations of the classification results, check the 'Results_Data_visualization.ipynb' notebook.
