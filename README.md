# CDS Project Outline
Clinical Decision Support System, an AI infrastructure of clinical decision making support, is dedicated to providing supportive advices, saving doctors from tedious and repeated work. The system includes the following sections: pathology detection -> result explaination -> similar cases analysis -> further prediction.

For the technical issues, different sections should be decoupled as mush as possible for the benefit of modular programming. **Because of the low requirements for concurrency performance, we serialize the execution of sections as presented before.**

## Pathology Detection (AI)
This section is the core of whole decision making system, the AI application receives the dicom images, transforms, preprocesses, makes segmentation and classifications. **Since there will be huge modification on this part, the rest of application should know nothing about it except for those predefined interfaces.**

Input: Dicom images for a single patients.

Output: 

1. The classification result.
2. PCA matrix, shape(n,2)
3. Saliency map with the same size of resized images.

## Result Explaination
After the inferencing section, this section will display the explaination of classification result and reveal clustering relationship among same groups from the PCA. Besides, there will also display the saliency map of each images for illustraing how AI give such a decision. 

Input:
1. The classification result.
2. PCA matrix, shape(n,2)
3. Saliency map with the same size of resized images.

Output: None

Side Effect: Display classification result(confidence on each class), PCA, saliency map.

## Similar Cases Analysis
This section implements KNN analysis based on previous PCA matrix and find out the 7 closest patients and revealing their diagnosis information in the dicom file header.

Input:
1. PCA matrix, shape(n,2)

Output: None

Side Effect: Implement KNN algorithm and display 7 closest patients information read from database.