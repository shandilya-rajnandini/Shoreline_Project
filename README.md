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
