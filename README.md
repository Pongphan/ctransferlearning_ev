**1.Title**

Enhanced object detection of *Enterobius vermicularis* eggs using cumulative transfer learning algorithm

**2.Project Description**

The methodology of this study can be delineated into several sequential steps, each contributing to the overall framework for the detection and classification of *Enterobius vermicularis* eggs. Code Information is a detailed, step-by-step description of the proposed method.

**3.Dataset Information**

*01_training_dataset* - the training dataset, images were meticulously cropped to a uniform dimension of 370 × 370 pixels, a process designed to preserve the salient morphological features of the parasitic eggs. A balanced dataset was established, consisting of 1000 images per class for a total of 2000 images. Class 0 comprises artifacts—objects or extraneous elements that may be erroneously interpreted as eggs yet lack diagnostic relevance. In contrast, Class 1 consists of *E. vermicularis* eggs, which are essential for the accurate diagnosis of enterobiasis.

*02_testing_dataset* - the images included in the testing dataset were meticulously cropped to dimensions of 370 × 370 pixels to ensure the preservation of critical morphological features of the parasitic eggs. The dataset comprised a total of 100 images, with 50 images allocated per class. Class 0 represented artifacts—non-parasitic structures or objects that may resemble eggs but are diagnostically irrelevant. Class 1 consisted of *E. vermicularis* eggs, which are the true parasitic elements essential for the accurate identification and diagnosis of enterobiasis.

*03_obj_det_original* - microscopic images are incorporated for object detection, as they capture high-resolution details critical for identifying and segmenting cellular structures.

**4.Code Information**

*ctransferlearning_ev.ipynb* a comprehensive framework for model training and testing in image classification, contrasting a conventional model with a proposed model. The conventional approach serves as a baseline, employing standard training procedures and established architectures, while the proposed model integrates innovative feature extraction and classification techniques. Both models were trained on a uniform dataset under rigorously controlled experimental conditions, and their performances were evaluated using a suite of metrics—including accuracy, precision, recall, and F1-score—to ensure a robust comparative analysis. The experimental outcomes indicate that the proposed model exhibits enhanced classification capabilities, thereby substantiating its potential as a viable alternative to traditional methods in image classification research.

*ObjectDetection_Rev2-Ev.ipynb* object detection was performed on microscopic images using a well-trained model. The detection process was quantitatively evaluated using the Intersection over Union (IoU) metric, which measures the overlap between the predicted object boundaries and the ground truth annotations.

**5.Usage Instructions**

1. Data Acquisition and Preprocessing
   The initial phase involves the importation of *E. vermicularis* egg images, which constitute the primary dataset for analysis. Given the inherent challenges associated with medical imaging datasets—such as limited sample size and variability—robust preprocessing is essential. The raw images undergo normalization and resizing to ensure consistency across the dataset, thereby facilitating subsequent processing stages.

2. Image Augmentation
   To enhance the diversity and robustness of the dataset, a comprehensive image augmentation pipeline is implemented. This process involves several techniques:
   - *Image Rotation:* Varying the orientation of the eggs to simulate different acquisition angles.  
   - *Gaussian Blur:* Applying a Gaussian filter to reduce noise and mimic various imaging conditions.  
   - *Gaussian Noise Addition:* Injecting synthetic noise to increase the model’s tolerance to real-world image variability.  
   - *Mean Filtering:* Smoothing the images to remove minor artifacts without significant loss of important features.  
   - *Image Sharpening:* Enhancing the edges and structural details of the eggs, thereby improving the model’s ability to distinguish fine morphological characteristics.  
   These augmentation strategies not only increase the effective size of the dataset but also improve the generalization ability of the deep learning models by simulating a broader range of imaging conditions.

3. Model Selection and Deep Learning Architectures
   The augmented dataset is subsequently used to train and evaluate multiple deep learning architectures. The study employs a combination of both conventional and state-of-the-art models:
   - *Convolutional Neural Network (CNN):* A baseline architecture to establish a performance benchmark.
   - *ResNet50:* Leveraging residual learning to address the degradation problem in deep networks.
   - *InceptionV3:* Utilizing multi-scale convolutional filters to capture diverse features.
   - *Denseet21:* Incorporating dense connectivity patterns to enhance feature propagation and reduce parameter redundancy.
   - *Xception:* Exploiting depthwise separable convolutions for improved efficiency and representational power.  
   Each model is fine-tuned on the augmented dataset, with hyperparameters optimized to maximize performance.

4. Cumulative Transfer Learning Algorithm
   Recognizing that different models excel in capturing various aspects of the image features, the study introduces a cumulative transfer learning algorithm. This algorithm operates by:
   - *Sequential Feature Integration:* Initially training a base model on the augmented dataset, followed by the extraction of its learned feature representations.
   - *Progressive Learning:* These features are then transferred and cumulatively integrated into subsequent models, which are further fine-tuned. This iterative process allows for the reinforcement of feature learning, as each model builds upon the previous one’s knowledge.
   - *Knowledge Aggregation:* The algorithm effectively aggregates the strengths of multiple architectures, resulting in a composite model that demonstrates improved performance over any individual network.  
   This cumulative approach ensures that the final model is robust, with enhanced capability in feature extraction and classification.

5. Performance Evaluation
   The final step involves a rigorous comparison of model performances using a variety of evaluation metrics:
   - *Image Classification:*
     - Confusion Matrix: To visualize and quantify the accuracy of predictions across different classes.
     - Area Under the Receiver Operating Characteristic Curve (AUC-ROC): To assess the trade-off between sensitivity and specificity, providing a comprehensive measure of the classifier’s performance.
   - *Object Detection:*
     - *Intersection over Union (IoU):* To evaluate the precision of object localization by comparing the overlap between the predicted and ground-truth bounding boxes.  
   These metrics facilitate a detailed quantitative analysis, allowing for an objective assessment of the efficacy of both the individual models and the cumulative transfer learning approach.

In summary, the proposed method integrates data augmentation, multiple deep learning architectures, and an innovative cumulative transfer learning algorithm to advance the state of the art in the detection and classification of *E. vermicularis* eggs. The method is further validated through robust performance metrics, ensuring its practical utility in both image classification and object detection tasks.

**6.Python Library Requirements**

cv2 4.11.0

numpy 2.0.2

matplotlib 3.10.0

sklearn 1.6.1

keras 3.8.0

**7.Citation**

This study used *E. vermicularis* egg images from the publicly available dataset, accessible through Figshare (https://doi.org/10.6084/m9.figshare.26266028.v2)  (Chaibutr N, Pongpanitanont P, Laymanivong S, Thanchomnang T, Janwan P. 2024. Development of a machine learning model for the classification of Enterobius vermicularis egg. Journal of Imaging 10(9):212 DOI 10.3390/jimaging10090212.)

**8.License**

This article is an open access article distributed under the terms and conditions of the Creative Commons Attribution (CC BY) license (https://creativecommons.org/licenses/by/4.0/).
