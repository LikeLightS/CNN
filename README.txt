This is a personal project on testing CNN (Convolutional Neural Network) on metal casting data to detect defects on the casting.
The data can be found in "https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product".

Files in this folder.
1_training.py - Trains the model with ResNet18 using the dataset provided to be specialized in detecting defects on metal casting.
2_predict_single.py - Using trained model to predict one images of the defected metal casting.
3_predict_test.py - Runs model testing on the trained model and output the performance/accuracy of the model on test dataset.
4_explain.py - Justification on the performance of the trained model.
5_web_app.py - Display the output onto a website using streamlit.

Conclusion
The 3_predict_test.py output shows the model accuracy score of 99.3%, correctly predict 710 images out of 715.
The 4_explain.py output shows the model correctly predicts using the right information; cracks, malformed, parts of the images as indicated by the red and orange region of the heatmap.
![Output from 4_explain.py showing heatmap over the image](output/explain_output.png)