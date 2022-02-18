# Multitask-Detection-System-Using-Deep-Learning-and-Flask

There are 3 modules in this project namely Covid-19 Detection, Drowsiness Detection and Face and Eye Detection. User can choose any one of the task among the given 3 tasks.

## Covid-19 Detection 
The user has to provide a chest x-ray image as an input and the model is trained on Kaggle dataset using CNN algorithm. The output is whether the person is infected or not.

## Drowsiness Detection
For Drowsiness detection, the model makes use of the webcam to detect the eye of the user. Initially the model is analyzed by viola jones to detect the face and eye of the user. Later the CNN algorithm acts upon it and once the person closes the eye for more than certain time (say 15 sec), the alarm starts up which alerts the user.

## Face & Eye Detection.
On choosing Face & Eye Detection, the user has to provide an image as the input and preprocessing of the image takes place to improve the quality of the image for further 
analyzing. The next step is feature extraction where the raw data is reduced to more manageable groups for processing. Finally, the model is trained using Voila Jones algorithm to detect the face and eye of the user (draws a square around the face and eye of the image)

More details of the project can be found in this paper: 
http://ymerdigital.com/uploads/YMER201300.pdf 
