# LGG-Segmentation-using-Brain-MRI-Scans

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
