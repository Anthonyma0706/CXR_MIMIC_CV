# Transfer Learning on MIMIC Chest X-Rays to Uncover Associations between Patient Health and Demographics

###
As advances in Artificial Intelligence (AI) become increasingly applied in healthcare settings,
information encoded in medical images such as Chest X-Rays (CXR) can be learned
and represented by Deep Learning models to make disease diagnoses and serve as a health
indicator. We designed this study to determine the relationships between social determinants
and health status captured by Chest X-Ray images as a data science solution to
explore social and health disparities. Specifically, we included two demographic features
(gender and race/ethnicity), and insurance coverage as a proxy measure to indicate socioeconomic
status. Using MIMIC-CXR dataset, we trained a Convolutional Neural Networks
(CNN) model to classify 1-year mortality as a way of learning the health information from
CXR images. Additional CNN models were trained independently for each feature that
can simultaneously serve as classification benchmarks trained on MIMIC-CXR cohort. To
interpret the model, we used Grad-CAM heatmaps to highlight regions that the model
mostly uses when making predictions, Grad-CAM heatmaps were then averaged and compared
across demographic models with 1-year mortality model. According to our analysis,
gender and race are the best predicted tasks (AUC 1.0, 0.92), and gender is less associated
with the health status information (1-year mortality model) compared to race and
insurance coverage.

<p align="center">
<img width="1003" alt="Screen Shot 2022-12-11 at 12 43 21 AM" src="https://user-images.githubusercontent.com/57332047/206888369-f439686b-157d-4080-8c55-2370804422e3.png">
<img width="978" alt="Screen Shot 2022-12-11 at 12 43 52 AM" src="https://user-images.githubusercontent.com/57332047/206888384-6dfe3296-1ddb-4544-8e09-f7686889dcae.png">


<img width="979" alt="Screen Shot 2022-12-11 at 12 45 44 AM" src="https://user-images.githubusercontent.com/57332047/206888441-7da0a6a1-1a07-4ac4-8b1a-2a203cc228fb.png">


<img width="690" alt="Screen Shot 2022-12-11 at 12 44 13 AM" src="https://user-images.githubusercontent.com/57332047/206888397-45f8c1cf-adec-4447-929b-7b4b8558c30f.png">
<img width="673" alt="Screen Shot 2022-12-11 at 12 45 11 AM" src="https://user-images.githubusercontent.com/57332047/206888431-37b26496-ec3d-414f-ad19-f7bcf9a318bf.png">
<img width="1040" alt="Screen Shot 2022-12-11 at 12 45 29 AM" src="https://user-images.githubusercontent.com/57332047/206888435-06c4f777-24d1-4e16-8ac7-a5f7924b14b9.png">

</p>
