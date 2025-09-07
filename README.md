# 🩺 Multi-View Mammogram Analysis with Segmented Breast Masses and Calcifications for BI-RADS Classification  

> 🏆 **Best Final Year Project Winner – Department of Electrical and Electronic Engineering, University of Sri Jayewardenepura, Sri Lanka**  

---

## 📌 Project Overview  
Breast cancer is the most prevalent cancer among women, and early detection plays a critical role in improving survival rates. Radiologists rely heavily on mammograms to detect abnormalities such as **breast masses** and **micro-calcifications**. However, manual interpretation is time-consuming, subject to fatigue, and prone to inter-observer variability.  

This project introduces a **multi-view deep learning framework** for the **pre-processing, segmentation, and classification** of mammograms into **BI-RADS categories**. The system integrates advanced enhancement pipelines, dual-path segmentation using **Modified Hybrid Transformer U-Net (HTU-Net)** and **U-Net**, and final classification with **EfficientNetB3 + ANN**. The proposed approach achieves high accuracy and robustness, with potential integration into **Computer-Aided Diagnosis (CAD) systems**.  

---

## 🧩 Methodology  

### 1. Pre-processing  
Pre-processing ensures **image standardization, enhancement, and ROI clarity**, preparing data for segmentation and classification.  

- **Dataset**: INBreast dataset (high-resolution DICOM mammograms with annotations).  
- **View Handling**: Cranio-Caudal (CC) and Mediolateral Oblique (MLO) views.  
- **Pectoral Muscle Removal**: Eliminated in MLO views to prevent misdiagnosis.  
![Pectoral_Muscle_Removal](https://drive.google.com/file/d/1taYXipFDwTgFXXrjTY0pAWjZvnofWfFw/view?usp=sharing)
- **Standardization**:  
  - Cropping breast tissue using largest external contour detection.  
  - Proportional resizing to **1024×1024 pixels**, preserving anatomical integrity.  
  - Zero-intensity padding to avoid distortions.  
![Standardization](https://drive.google.com/file/d/1dKJihOHzCAYKL6Z4bEg1WF5DExiLHPP6/view?usp=sharing)

#### Mass Enhancement Pipeline  
- Contrast Limited Adaptive Histogram Equalization (**CLAHE**).  
- Global thresholding to highlight suspicious ROIs.  
- Magma color mapping.  
- Channel decomposition & fusion (R-G, R-B, B-G).  
- **B-G fusion** achieved the best performance (CNR, SSIM, Colorfulness).  
![Mass_Enhancement_Pipeline ](https://drive.google.com/file/d/1Ui4RYMobHxZdj_AQ6xuHb4VnFye6x_8r/view?usp=sharing)
![Mass_Enhancement_Pipeline2](https://drive.google.com/file/d/1NaGiep3OB8yQjJQPWyj5nhZEa59q9Siv/view?usp=sharing)

#### Calcification Enhancement Pipeline  
- High-pass filtering (Gaussian low-pass subtraction).  
- Rectification + Gaussian smoothing to improve SNR.  
- Adaptive thresholding to isolate calcifications.  
- Overlaying binary mask on mammogram for improved visualization.  
![Calcification_Enhancement_Pipeline2](https://drive.google.com/file/d/1AJCiQBjBvZPK0UYmExgLvWrVF1qnn6Pf/view?usp=sharing)

✅ *Results:* Up to **72% improvement in CNR for masses** and **46% improvement in PSNR for calcifications** compared to conventional methods.  

---

### 2. Segmentation  
A **dual-path segmentation framework** was developed to handle differences between **masses** (irregular, diffuse) and **calcifications** (tiny, high-intensity).  

#### Dual-Path Framework  
- **Mass segmentation** → Modified **HTU-Net**  
- **Calcification segmentation** → Standard **U-Net with patch-based reconstruction**  
- **Fusion** → Overlay of results with distinct colors for clarity.  



#### Mass Segmentation (Modified HTU-Net)  
- Encoder-decoder structure enhanced with **Spatial-Channel Enhanced Self Attention (SCESA)**.  
- Multi-Head Attention (MHA) at bottleneck for global dependencies.  
- Conv2D positional encodings instead of patch embeddings.  
- **Performance**: DSC = **0.6064**, Precision = **0.5988**.  
![Mass_Segmentation](https://drive.google.com/file/d/1JP9KiDygrBZh_K8XMwfQedn2QvtW5VBN/view?usp=sharing)

#### Calcification Segmentation (U-Net + Patch-based Method)  
- 64×64 patches (stride 32), reconstructed with cosine Hanning window merging.  
- Encoder: 3 convolutional blocks; Bottleneck: 512 filters; Decoder: 3 upsampling blocks.  
- **Performance**: DSC = **0.8022**, Precision = **0.8958**.  
![Calcification_Segmentation](https://drive.google.com/file/d/1XD_DSKjU8NEx_ELFjeR7mBMIDZfj1e8M/view?usp=sharing)

#### Post-processing  
- Thresholding → Binary mask.  
- Morphological closing → Smooth boundaries, remove noise.  
- Largest connected component retained (for masses).  
- Boundary refinement for calcifications by multiplying masks with normalized pre-processed images.  
- Final overlay of masses + calcifications.  
![Post-processing ](https://drive.google.com/file/d/1Kivk1SqOwhUGqNDzDm1yzSSetf-uzBw5/view?usp=sharing)

#### Results 
![Results ](https://drive.google.com/file/d/1s9Q9aZhQb3fumTLkX73HsDCgfkLAtETB/view?usp=sharing)
![Results](https://drive.google.com/file/d/1-tQzg544tnaazj3Z3DlhPfuaO6Ue2UgI/view?usp=sharing)
---

### 3. Classification  
After segmentation, ROIs are classified into **BI-RADS categories (2–5)**.  

#### Models Evaluated  
- **Single View CNN** (baseline).  
- **Multi-View CNN** (parallel CC + MLO pathways).  
- **EfficientNetB3 + ANN (Proposed)**:  
  - Pre-trained on ImageNet, fine-tuned with mammograms.  
  - Global average pooling + feature concatenation across both views.  
  - Dense ANN layers as classifier.  
  - Optimizer: Adam, LR = 0.0005, with 5-fold cross-validation.  
![Results](https://drive.google.com/file/d/1bcIX3B8klyv_dxS-tybpSNCeSm1uSZf4/view?usp=sharing)

#### Class Imbalance Handling  
- Enhanced **SMOTE with adaptive k-neighbors** to balance BI-RADS categories.  

#### Results  
![Results](https://drive.google.com/file/d/1yBmW8EfelatJL0zc_93NFeaJcVlShsFN/view?usp=sharing)

---

## 🔬 Key Contributions  
1. **Dual-path enhancement pipelines** tailored for masses and calcifications.  
2. **Segmentation framework** with modified HTU-Net and U-Net for accurate ROI detection.  
3. **Multi-view classification pipeline** with EfficientNetB3 + ANN for BI-RADS categorization.  
4. Achieved **state-of-the-art performance** compared to conventional methods.  

---

## 🚀 Future Work  
- Extend to **multi-institutional datasets** for generalizability.  
- Add **explainability modules** (e.g., Grad-CAM).  
- Real-time deployment for clinical use.  
- Expand classification to include **benign vs malignant predictions**.  

---

## 📄 Publications  

1. **Dual-Path Enhancement Framework for Masses and Calcifications in Mammograms: A Quantitative Evaluation of Preprocessing Techniques**  
   - *Authors:* **S.A.A.J.M. Athukorala, S.M.D. Deepashika, T.D. Wickramasingha, I.A.S.I. Fernando, U.L. Wijewardhana, U.B. Balagalla**  
   - *Conference:* [Insert Conference Name & Year — already held]  
   - *Status:* ✅ Accepted & Presented  
   - *Summary:* Introduced a **dual-path preprocessing pipeline**, achieving major improvements in CNR and SSIM.  

2. **A Unified Deep Learning Approach for the Segmentation of Breast Masses and Calcifications**  
   - *Authors:* **T.D. Wickramasingha, S.A.A.J.M. Athukorala, S.M.D. Deepashika, I.A.S.I. Fernando, U.L. Wijewardhana, U.B. Balagalla**  
   - *Conference:* [Insert Conference Name & Year — upcoming]  
   - *Status:* ✅ Accepted (Presentation Scheduled)  
   - *Summary:* Proposed **Modified HTU-Net** for masses and **U-Net** for calcifications, showing strong segmentation accuracy.  

3. **Unified Multi-View Mammogram Analysis: Segmentation and BI-RADS Categorization of Masses and Calcifications**  
   - *Authors:* **S.A.A.J.M. Athukorala, S.M.D. Deepashika, T.D. Wickramasingha**  
   - *Journal:* [Insert Journal Name]  
   - *Status:* ⏳ Under Review  
   - *Summary:* End-to-end pipeline integrating preprocessing, segmentation, and classification, achieving **87.55% accuracy**.  

---

## 🏆 Achievements  

- 🥇 **Best Final Year Project Award**  
  - Awarded by the **Department of Electrical and Electronic Engineering, University of Sri Jayewardenepura, Sri Lanka**.  
  - Recognized for outstanding innovation, research contribution, and successful integration of deep learning into clinical support systems.  
![Results](https://drive.google.com/file/d/1O2Jy0oGnJn329sUDWNPAchIqmgBd8ywE/view?usp=sharing)
---
