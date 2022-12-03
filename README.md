# Deep Learning based LGG Segmentation using Brain MRI Scans

Motivation: Low-grade gliomas (LGGs) are a diverse group of primary brain tumors that often arise in young, otherwise healthy patients and generally have an indolent course with longer-term survival in comparison with high-grade gliomas. Treatment options include observation, surgery, radiation, chemotherapy, or a combined approach, and management is individualized based on tumor location, histology, molecular profile, and patient characteristics. Moreover, in this type of brain tumor with a relatively good prognosis and prolonged survival, the potential benefits of treatment must be carefully weighed against potential treatment-related risks.
Models for automated segmentation of Low Grade Gliomas in brain MRI scans

Aim: To create Deep Learning based models that perform automated segmentation of LGGs in Brain MRI scans

Dataset: TCGA-LGG from Kaggle --> https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation
Contains 3939 images and corresponding masks

Models Tested:
  1. Vanilla UNet
  2. UNet + Batch Normalization
  3. Alphax-UNet - Reduced UNet for lighter models
  4. Recurrent-Residual UNet (R2 UNet)
  
Results:
![image](https://user-images.githubusercontent.com/65041568/205439541-c8e66f70-e60f-4d0e-9815-5007b2e71689.png)

Inferences:
1. MR scan inference is about 95% accurate  and the best model gave 91% F1 score
2. MRI takes about 30 minutes and results by clinicians take about 1 day (for segmentation and analysis). Although inference time for the R2-UNet was about 22s, the model would not indicate anything beyond the segmentation like type or severity of tumor unless specifically trained
3. The model can be easily deployed on radiology computers or hand-held tablets as the model takes about 350 MB of storage (.hdf5 format). It can be deployed easily using platforms like Docker or Vertex AI.
4. From the given dataset of 3939 MR scans, the results were promising. Adding more images with different types of tumors could be a next logical step to build a more expansive models that works on cases beyond LGG.
5. Experimentations Performed:
  Adding batch normalization to vanilla UNet as it gave better performance. Changing number of kernels in the UNet to check performance on lighter models. Using decaying learning rate, which did not yield any better results than what fixed rates gave. Data augmentation with only horizontal flips in order to maintain ethos of brain MRI.
  
  References:
  1. Buda, Mateusz, Ashirbani Saha, and Maciej A. Mazurowski. "Association of genomic subtypes of lower-grade gliomas with shape features automatically extracted by a deep learning algorithm." Computers in biology and medicine 109 (2019): 218-225.
  2. Chen, Min, et al. "Deep feature learning for medical image analysis with convolutional autoencoder neural network." IEEE Transactions on Big Data 7.4 (2017): 750-758.
  3. Kugelman, J., Allman, J., Read, S.A. et al. A comparison of deep learning U-Net architectures for posterior segment OCT retinal layer segmentation. Sci Rep 12, 14888 (2022). https://doi.org/10.1038/s41598-022-18646-2
  4. Alom, Md Zahangir, et al. "Nuclei segmentation with recurrent residual convolutional neural networks based U-Net (R2U-Net)." NAECON 2018-IEEE National Aerospace and Electronics Conference. IEEE, 2018.

