# Shoreline Project 
 
This is a Streamlit-based web application. 
 
## How to Run the Project Locally 
 
### 1. Clone the repository 
```bash 
git clone https://github.com 
cd Shoreline_Project 
``` 
 
### 2. Setup Virtual Environment 
```bash 
python -m venv venv 
venv\Scripts\activate 
``` 
 
### 3. Install Dependencies 
```bash 
pip install -r requirements.txt 
``` 
 
### 4. Run the Application 
```bash 
streamlit run app.py 
``` 
 
## Terrain-Adaptive Processing Strategy 
 
To handle different geomorphological coastal features, use the following configuration guidelines: 
 
| Coastal Terrain | Denoising Filter | Segmentation Model | Physics-Based Rationale | 
| :--- | :--- | :--- | :--- | 
| **Rocky / High-Relief Cliffs** | Adaptive Lee / Refined Lee | Tiled Otsu | Sharp edge contrast; preserves steep boundaries without geometric displacement. | 
| **Open Sandy Beaches** | Lee / Kuan Filter | Otsu / K-Means | Clean bimodal separation between dry sand and sea; limits speckle noise. | 
| **Mangrove / Estuaries** | NL-Means / Lee Filter | Fuzzy C-Means (FCM) / K-Means | Heavy canopy penetration requires soft membership probabilities for mixed pixels. | 
| **Intertidal Mudflats / Marshes** | Non-Local Means (NL-Means) | FCM / Tiled Otsu | Mitigates specular reflections on low-contrast, wet exposed surfaces. | 
| **Urban Zones / Harbors** | Lee Filter | K-Means + CCA Purification | Rejects bright man-made reflectors and isolated non-coastal artifacts like ships. | 
 
## Metrics and Analytics Performance 
 
The framework's **Purified Analytics Matrix** delivers sub-pixel precision across long swaths: 
- **Positional Accuracy:** achieves an RMSE of ñ8-14 meters compared against optical validation sets. 
- **Noise Reduction:** Connected Component Analysis (CCA) reduces the false-positive land misclassification rate to under 2.5%. 
- **Vector Optimization:** Douglas-Peucker algorithmic compression compresses redundant polyline vertices by approximately 65% for immediate GIS portability. 
 
## ?? Project Documentation & Publications 
 
Read the full detailed machine learning framework, system architecture, and validation protocol here: 
- [Download Research Paper PDF](./NISAR_Shoreline_Extraction_Paper.pdf) 
 
## ?? Connect and Reach Me 
 
If you are interested in my work on radar remote sensing, machine learning, or coastal monitoring frameworks, feel free to connect! 
- **Email:** singhrajnandini708@gmail.com 
- **GitHub Profile:** [shandilya-rajnandini](https://github.com/shandilya-rajnandini) 
- **Institution:** Department of Computer Science and Engineering, IIT Patna 
 
## ?? Processed NISAR Dataset Catalogue (Cycle 25 / L25) 
 
The framework has been successfully validated on Level-2 Geocoded Covariance **(L2-GCOV)** dual-frequency files spanning **Cycle 25 (L25)**. Below are the core datasets processed and catalogued in this repository: 
 
- **`NISAR_L2_PR_GCOV_025_005_D_082_4005_DHDH_A_20260708T132740_20260708T132815_P05023_N_F_J_001.h5`** 
- **`NISAR_L2_PR_GCOV_025_005_D_083_4005_DHDH_A_20260708T132814_20260708T132833_P05023_N_P_J_001.h5`** 
- **`NISAR_L2_PR_GCOV_025_034_D_078_4005_DHDH_A_20260710T134205_20260710T134240_P05023_N_F_J_001.h5`** 
- **`NISAR_L2_PR_GCOV_025_034_D_079_4005_DHDH_A_20260710T134239_20260710T134313_P05023_N_F_J_001.h5`** 
- **`NISAR_L2_PR_GCOV_025_034_D_080_4005_DHDH_A_20260710T134312_20260710T134333_P05023_N_P_J_001.h5`** 
- **`NISAR_L2_PR_GCOV_025_048_D_079_4005_DHDH_A_20260711T130102_20260711T130136_P05023_N_F_J_001.h5`** 
- **`NISAR_L2_PR_GCOV_025_048_D_080_2005_DVDV_A_20260711T130152_20260711T130210_P05023_N_P_J_001.h5`** 
- **`NISAR_L2_PR_GCOV_025_048_D_080_4005_DHDH_A_20260711T130135_20260711T130149_P05023_N_P_J_001.h5`** 
- **`NISAR_L2_PR_GCOV_025_077_D_083_4005_DHDH_A_20260713T131954_20260713T132029_P05023_N_F_J_001.h5`** 
- **`NISAR_L2_PR_GCOV_025_077_D_084_4005_DHDH_A_20260713T132028_20260713T132100_P05023_N_F_J_001.h5`** 
- **`NISAR_L2_PR_GCOV_025_084_A_007_4005_DHDH_A_20260714T001555_20260714T001630_P05023_F_F_J_001.h5`** 
- **`NISAR_L2_PR_GCOV_025_149_D_083_4005_DHDH_A_20260718T131135_20260718T131210_P05023_N_F_J_001.h5`** 
- **`NISAR_L2_PR_GCOV_025_098_A_012_4005_DHDH_A_20260714T233715_20260714T233740_P05023_F_P_J_001.h5`** 
- **`NISAR_L2_PR_GCOV_025_106_D_081_4005_DHDH_A_20260715T133526_20260715T133604_P05023_N_F_J_001.h5`** 
- **`NISAR_L2_PR_GCOV_025_156_A_009_4005_DHDH_A_20260719T000843_20260719T000917_P05023_N_F_J_001.h5`** 
- **`NISAR_L2_PR_GCOV_025_149_D_084_4005_DHDH_A_20260718T131209_20260718T131240_P05023_N_F_J_001.h5`** 
- **`NISAR_L2_PR_GCOV_025_149_D_082_4005_DHDH_A_20260718T131105_20260718T131136_P05023_N_F_J_001.h5`** 
- **`NISAR_L2_PR_GCOV_025_135_D_078_2005_DVDV_A_20260717T135043_20260717T135059_P05023_N_P_J_001.h5`** 
- **`NISAR_L2_PR_GCOV_025_120_D_079_4005_DHDH_A_20260716T125242_20260716T125300_P05023_N_P_J_001.h5`** 
