<a name="br1"></a> 
<img width="179" alt="image" src="https://github.com/sdarwish13/FaceEmotionRecognition/assets/66074964/d0bee029-57fe-4ff1-89a9-bc80fb74e3ff">

<p align="left">
**CMPS 287 Artificial Intelligence - Spring 2020-2021**

**Project Report**

</p>
<p align="center">
  ***Hadi Najdi** 201804070*

  ***Sara Darwish** 201902020*

  ***Karim Ghaddar** 201903120*
</p>

Project Abstract:

The aim of our project is to create a program that can identify human emotions through pictures.

We want to create a method of constructing an emotion recognition system based on a dataset,

which includes an optimized algorithm for generating training and test samples which help us

identify the human emotion. This project is devoted to the optimization of the recognition

method of seven basic emotions (joy, sadness, fear, anger, surprise, disgust and neutral) in terms

of the expressions of the human face. This method focuses on the several criteria of the person’s

face such as eyes, mouth, and eyebrows.

Introduction & Dataset:

**Dataset Description:**

The FER-2013 dataset consists of 28,000 labelled images in the training set, 3,500 labelled

images in the development set, and 3,500 images in the test set. The dataset was created by

gathering the results of a Google image search of each emotion and synonyms of the emotions.

Each image in FER-2013 is labelled as one of seven emotions, such as happy, sad, angry, afraid,



<a name="br2"></a> 

surprise, disgust, and neutral, with happy being the most prevalent emotion, providing a baseline

for random guessing of 24.4%.

<https://www.kaggle.com/deadskull7/fer2013>

**Problem:**

Face detection is a pre-processing phase to recognize facial expression of human, Second Train

our Algorithm from our dataset then we need to test it. We will separate our dataset into training,

validation, and testing. Although it can be trained to detect a variety of object classes, it was

motivated primarily by the problem of face detection. Despite this variety, face recognition faces

some issues inherent to the problem definition, environmental conditions, and hardware

constraints. In terms of side Face or where the image is not providing proper frontal face view,

these algorithms and methods give very less accurate recognition of emotion.

Methodology and Related Work:

After thorough research, we have decided that the best model to use for our project is the deep

neural network model. We read many articles that showed that the most efficient model to use

and the model that, after training, gives the best outcomes is neural networks.

<https://www.sciencedirect.com/science/article/pii/S235291482030201X>

Real-time emotion recognition has been an active field of research over the past several decades. This

work aims to classify physically disabled people (deaf, dumb, and bedridden) and Autism children's

emotional expressions based on facial landmarks and electroencephalograph (EEG) signals using a

convolutional neural network (CNN) and long short-term memory (LSTM) classifiers by developing an

algorithm for real-time emotion recognition using virtual markers through an optical flow algorithm that

works effectively in uneven lightning and subject head rotation (up to 25°), different backgrounds, and

various skin tones.



<a name="br3"></a> 

<https://link.springer.com/article/10.1007/s11042-020-09405-4>

Emotions represent a key aspect of human life and behaviour. In recent years, automatic recognition of

emotions has become an important component in the fields of affective computing and human-machine

interaction. The creation of a generalized, inter-subject, model for emotion recognition from facial

expression is still a challenge, due to anatomical, cultural, and environmental differences. On the other

hand, using traditional machine learning approaches to create a subject-customized, personal, model

would require a large dataset of labelled samples. For these reasons, in this work, we propose the use of

transfer learning to produce subject-specific models for extracting the emotional content of facial images

in the valence/arousal dimensions. Transfer learning allows us to reuse the knowledge assimilated from a

large multi-subject dataset by a deep-convolutional neural network and employ the feature extraction

capability in the single subject scenario.

<https://www.sciencedirect.com/science/article/pii/S1877050917305264>

In the article there are presented the results of recognition of seven emotional states (neutral, joy, sadness,

surprise, anger, fear, disgust) based on facial expressions. Coefficients describing elements of facial

expressions, registered for six subjects, were used as features. The features have been calculated for three-

dimensional face model. The classification of features was performed using k-NN classifier and MLP

neural network.

Model:

Our model contains one input layer and six hidden layers, one of them is dense and one output

layer. We chose Exponential Linear Unit (elu) as our activation function and added

EarlyStopping and ReduceLROnPlateau as callback functions for our model fitting.



<a name="br4"></a> 

Experimentation and Results:

When experimenting with the hyper parameters of our model, we first chose the number of

layers and nodes and dropout values by trial and error while choosing the batch size and initial

learning rate and decay by running a random search on our model. We got the random search

results according to the best accuracy score and trained our model with those parameters and got

a validation accuracy of 0.69 and validation loss of 0.86. The early stopping was based on the

accuracy and not the loss and so we believe we could have gotten a better loss if the early

stopping was based on a minimal loss.

Loss Graph:

![image](https://github.com/sdarwish13/FaceEmotionRecognition/assets/66074964/082c5016-def8-40ad-b00c-a4ecd6fd59f1)


Accuracy Graph:

![image](https://github.com/sdarwish13/FaceEmotionRecognition/assets/66074964/40755ac1-2745-41ed-aae5-a97a172edcdf)


<a name="br5"></a> 

According to the learning curves above, we notice that there is no overfitting or underfitting in

our model.

The results we got were acceptable, but we could have obtained better ones if we had a larger

and better dataset. Our Model could be starting step, for many of the emotion-based applications

such as lie detector and, also mood-based learning for students.
