import numpy as np
from preprocessing.multilooking import apply_multilooking
from denoising.lee_filter import apply_lee_filter
from segmentation.otsu_threshold import OtsuSegmenter
from segmentation.kmeans_clustering import KMeansSegmenter
from refinement.cca_refinement import RefinementStep
from refinement.canny_edges import extract_canny_vector

class UltimateShorelinePipeline:
    def __init__(self, raw_sar_data):
        self.raw_data = raw_sar_data
        
    def execute_pipeline(self, denoise_method='lee', segment_method='otsu'):
        current_matrix = apply_multilooking(self.raw_data, look_window=(2,2))
        if denoise_method == 'lee':
            current_matrix = apply_lee_filter(current_matrix)
        if segment_method == 'otsu':
            engine = OtsuSegmenter()
        elif segment_method == 'kmeans':
            engine = KMeansSegmenter()
        else:
            engine = OtsuSegmenter()
            
        binary_mask = engine.segment(current_matrix)
        refiner = RefinementStep()
        clean_mask = refiner.refine(binary_mask)
        return extract_canny_vector(clean_mask)

if __name__ == "__main__":
    fake_scene = np.random.uniform(0, 1, (100, 100))
    pipeline = UltimateShorelinePipeline(fake_scene)
    result = pipeline.execute_pipeline(denoise_method='lee', segment_method='kmeans')
    print("====== PIPELINE STATUS ======")
    print("🚀 Shoreline Project Folder System Windows par full code ke saath ready hai!")
    print("Output Vector Shape:", result.shape)
