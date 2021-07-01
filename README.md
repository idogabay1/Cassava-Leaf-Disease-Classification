# Cassava-Leaf-Disease-Classification
In this project we will classify the diseases of the Cassava Leaf to 5 categories. We will train on imbalanced train-set but measure he result on balanced test-set
Pytorch implementation.
Based on Cassava Leaf Disease Classification [dataset from Kaggle](https://www.kaggle.com/c/cassava-leaf-disease-classification)
you can find our presentation on [Youtube](https://youtu.be/yg20D6vt6BA)

- [Cassava-Leaf-Disease-Classification](#Cassava-Leaf-Disease-Classification)
  * [Background](#Background)
  * [Files in the repository](#Files-in-the-repository)
  * [References](#References)
  
![3852521](https://user-images.githubusercontent.com/81647059/124014048-b27cd700-d9eb-11eb-8776-601fa6e6ef60.jpg)
![177414500](https://user-images.githubusercontent.com/81647059/124014120-ca545b00-d9eb-11eb-9947-1962fc512d6b.jpg)
![205418485 jpg_copy3](https://user-images.githubusercontent.com/81647059/124014153-d50ef000-d9eb-11eb-8794-4c58ccc567ab.jpg)

  
  
  
  
  
# Background
Cassava is the second largest provider of carbohydrates in Africa.  
its Viral diseases are major sources of poor yields

This Kaggle Challange try to classify 21398 images of Cassava plant leafs into 5 classes of diseases:  
  0 - Cassava Bacterial Blight (CBB)  
  1 - Cassava Brown Streak Disease (CBSD)  
  2 - Cassava Green Mottle (CGM)  
  3 - Cassava Mosaic Disease (CMD)  
  4 - Healthy  
  
After investigationg a little deeper (pun intended), we decided to solve similar but yet different route :  
overcome the imbalanced train-set over balanced test-set.  
62% of the Cassava Leaf Disease Classification Dataset labeled as class 3 (CMD).  
we decided to try train our models on train-set with the same class distribution as the whole dataset but measure our results on balanced validation and test sets.  
to do so we tried few Deep Learning architectures including originals CNN and Resnet18.  
we used google colab to run our code.


# Files in the repository

Both file are well documented. just follow the documentation and you will be fine :)

|File name         | Description |
|----------------------|------|
|`Cassava Leaf Disease Classification on balanced test-set.ipynb`| the main file in Google Colab format. to open import to Google Colab|
|`Cassava Leaf Disease Classification on balanced test-set.py`| the main file in Python format|

# References

Dataset source: https://www.kaggle.com/c/cassava-leaf-disease-classification  
Fork of cassava model: https://www.kaggle.com/charlesrongione/fork-of-cassava  
spatial transformer network implementation: https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html  
Optuna: https://optuna.org    
stochastic depth implementaion: https://link.springer.com/chapter/10.1007/978-3-319-46493-0_39  
