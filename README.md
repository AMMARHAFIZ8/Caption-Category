# Caption-Category

Categorize caption into 5 categories of articles into 5 categories namely Sport, Tech, Business, Entertainment and politics. 


### EDA 
## 1) Data Loading

## 2) Data Inspection

## 3) Data Cleaning

## 4) Features Selection
Nothing to select

## 5) Preprocessing
  
  1) Convert into lower case
  2) Tokenization
  3) Padding & truncating
  4) One hot encoding
  5) Train test split
  
  

### Model Development

1) LSTM Bidirectional Embedding
2) Stopping Callbacks 
3) Tensorboard
4) Plot Visualisation


### Model Evaluation / Analysis

1) accuracy score
2) F1 score 

f1, Accuracy score = 89%
![Alt text](https://github.com/AMMARHAFIZ8/Caption-Category/blob/main/acc.PNG)

Model

![Alt text](https://github.com/AMMARHAFIZ8/Caption-Category/blob/main/plot%20and%20graph/model.png)

traning accuracy , validation accuracy

![Alt text](https://github.com/AMMARHAFIZ8/Caption-Category/blob/main/plot%20and%20graph/Figure%202022-06-23%20235112-%20traning%20validation%20acc.png)

traning loss , validation loss

![Alt text](https://github.com/AMMARHAFIZ8/Caption-Category/blob/main/plot%20and%20graph/Figure%202022-06-23%20235211%20train%20val%20loss.png)

Tensorboard

![Alt text](https://github.com/AMMARHAFIZ8/Caption-Category/blob/main/plot%20and%20graph/Tensorboard.PNG)

![Alt text](https://github.com/AMMARHAFIZ8/Caption-Category/blob/main/plot%20and%20graph/Tensorboard-time-series.PNG)
 
![Alt text](https://github.com/AMMARHAFIZ8/Caption-Category/blob/main/plot%20and%20graph/train.png)


 ### Discussion/Reporting

The accuracy score and f1 of this model is 88.56% score
Model is consider great and its learning from the training.  
Training graph shows an overfitting since the training accuracy is higher  than validation accuracy
     
This model seems not give any effect although Earlystopping with LSTM can overcome overfitting.
With suggestion to overcome overfitting can try other 
architecture like BERT, transformer or GPT3 model.




[Raw Data  ]([http://archive.ics.uci.edu/ml/datasets/Heart+Disease](https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv))
