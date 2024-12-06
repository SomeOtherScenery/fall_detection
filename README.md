# A Real-time Skeleton-based Fall Detection Algorithm based on Temporal Convolutional Networks and Transformer Encoder Environment

## Environment
* python 3.10
*	torch 2.1.1 + cu118
*	numpy 1.22.4

## Script

*	dataset.py: arrange the dataset (normalization) for training 
*	loss.py: focal loss 
*	network.py: includes Temporal Convolutional Network (TCN), TransformerEncoder (TE), and combined TCNTE
*	train_tcn_classifier_1.py: TCN training
*	train_transformer_classifier_1.py: TEtraining
*	train_tcn_transformer_1.py: TCNTE training

## Demo video (based on UP-Fall dataset)

*	YOLOv8 + BoT-SORT + TCNTE model
*	Platform: NVIDIA Jetson Orin NX
  
<video width="600" controls>
  <source src="fall_detection/example.mp4" type="video/mp4">
</video>

## Reference(preprint) 
Yu, Xiaoqun and Wang, Chenfeng and Wu, Wenyu and Xiong, Shuping, An Efficient Skeleton-Based Fall Detection Algorithm Using Temporal Convolutional Networks with Transformer Encoder. Available at SSRN: https://ssrn.com/abstract=4750350 or http://dx.doi.org/10.2139/ssrn.4750350

--------------------------------------------------------------------
**More detailed documentation and the formal publication will be available soon… Please feel free to raise an issue if you have any questions.**
