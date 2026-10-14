import os
import io
import json
import zipfile
import tempfile
import numpy as np
import cv2
import h5py
import geopandas as gpd
from shapely.geometry import LineString
from scipy.ndimage import uniform_filter, label as scipy_label, binary_erosion
from skimage.filters import threshold_otsu
from skimage.measure import regionprops
from skimage.morphology import closing, opening, disk
from pyproj import CRS, Transformer


def find_nisar_main_path(f):
    for band in ['LSAR', 'SSAR']:
        path = f"science/{band}/GCOV/grids/frequencyA"
        if path in f:
            pols = list(f[path].keys())
            if pols:
                return f"{path}/{pols[0]}"
    return None


def get_nisar_metadata(file_path):
    with h5py.File(file_path, 'r') as f:
        target = find_nisar_main_path(f)
        if not target:
            return None
        ds = f[target]
        return {
            "height": int(ds.shape[-2]),
            "width": int(ds.shape[-1]),
            "spacing": float(ds.attrs.get('pixelSpacingX', 10.0)),
            "band": "L-Band" if "LSAR" in target else "S-Band"
        }


def get_nisar_data(file_path):
    with h5py.File(file_path, 'r') as h5_file:
        target = find_nisar_main_path(h5_file)
        if not target:
            raise ValueError("Invalid NISAR HDF5 structure.")
        ds = h5_file[target]
        data = ds[0, ::6, ::6] if ds.ndim == 3 else ds[::6, ::6]
        return np.abs(data)**2 if data.dtype in [np.complex64, np.complex128] else np.array(data, dtype=np.float32)


def apply_log_scaling(intensity):
    mask = np.isfinite(intensity) & (intensity > 0)
    db = np.full(intensity.shape, -30.0, dtype=np.float32)
    db[mask] = 10 * np.log10(intensity[mask])
    db_clipped = np.clip(db, -30.0, 5.0)
    viz = cv2.normalize(db_clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return db_clipped, viz, mask


def fuse_nisar_bands(file_l, file_s, method="PCA"):
    data_l = get_nisar_data(file_l)
    data_s = get_nisar_data(file_s)
    if data_l.shape != data_s.shape:
        data_s = cv2.resize(data_s, (data_l.shape[1], data_l.shape[0]), interpolation=cv2.INTER_LINEAR)

    db_l, _, m_l = apply_log_scaling(data_l)
    db_s, _, m_s = apply_log_scaling(data_s)
    fused_mask = m_l & m_s

    if method == "PCA":
        l_flat = cv2.normalize(db_l, None, 0, 1, cv2.NORM_MINMAX).flatten()
        s_flat = cv2.normalize(db_s, None, 0, 1, cv2.NORM_MINMAX).flatten()
        X = np.vstack((l_flat, s_flat))
        cov = np.cov(X)
        val, vec = np.linalg.eig(cov)
        w = vec[:, np.argmax(val)]
        w = w / np.sum(w)
        fused_db = (w[0] * db_l) + (w[1] * db_s)
    else:
        fused_db = (db_l + db_s) / 2.0

    viz = cv2.normalize(np.clip(fused_db, -30.0, 5.0), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return fused_db, viz, fused_mask


def run_speckle_filter(img, method):
    if method == "Lee Filter":
        m = uniform_filter(img, 5)
        v = uniform_filter(img**2, 5) - (m**2)
        w = v / (v + np.var(img) + 1e-5)
        res = m + w * (img - m)
    elif method == "Kuan Filter":
        m = uniform_filter(img, 5)
        v = uniform_filter(img**2, 5) - (m**2)
        ci = v / (m**2 + 1e-5)
        w = np.clip((1 - 0.06 / (ci + 1e-5)) / 1.06, 0, 1)
        res = m + w * (img - m)
    else:
        res = cv2.medianBlur(img.astype(np.float32), 5)

    viz = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    enl = round(float((np.mean(res) / (np.std(res) + 1e-5))**2), 2)
    return res, viz, enl


def run_segmentation(img, method, mask, manual_t=None):
    if manual_t is not None:
        t_val = manual_t
    elif method == "K-Means Clustering":
        pixel_values = img[mask].reshape((-1, 1))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixel_values.astype(np.float32), 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        t_val = np.mean(centers)
    else:
        t_val = threshold_otsu(img[mask])

    bin_img = (img < t_val) & mask
    if np.sum(bin_img) > np.sum(mask) * 0.6:
        bin_img = ~bin_img & mask
    return bin_img.astype(np.uint8) * 255, round(float(t_val), 2)


def run_refinement(binary, r_list, spacing, swath_mask, raw_db, morph_val=21):
    bin_u8 = (binary > 0).astype(np.uint8) * 255
    res = cv2.morphologyEx(bin_u8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_val, morph_val)))
    lab, num = scipy_label(res > 0)
    if num == 0:
        return np.zeros_like(binary), {"KM": 0, "Area": 0}, bin_u8

    ocean_mask = (lab == max(regionprops(lab), key=lambda r: r.area).label).astype(np.uint8) * 255
    purified = opening(closing(ocean_mask > 0, disk(3)), disk(11))
    edges = cv2.Canny(purified.astype(np.uint8) * 255, 100, 200)
    m_mask = binary_erosion(swath_mask, structure=np.ones((40, 40)))
    clean_edges = cv2.bitwise_and(edges, edges, mask=m_mask.astype(np.uint8))
    cnts, _ = cv2.findContours(clean_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    final_canvas = np.zeros_like(edges)
    total_km = 0
    if cnts:
        main_cnt = max(cnts, key=lambda c: cv2.arcLength(c, False))
        if "Douglas-Peuker" in r_list:
            epsilon = 0.0015 * cv2.arcLength(main_cnt, True)
            main_cnt = cv2.approxPolyDP(main_cnt, epsilon, False)
        total_km = (cv2.arcLength(main_cnt, False) * spacing) / 2000.0
        cv2.drawContours(final_canvas, [main_cnt], -1, 255, 1)

    area = round((np.sum(purified > 0) * (spacing**2)) / 1e6, 2)
    return final_canvas, {"KM": round(total_km, 2), "Area": area}, purified.astype(np.uint8) * 255


def calculate_change(m1, m2, sp):
    er = (m1 > 0) & (m2 == 0)
    ac = (m1 == 0) & (m2 > 0)
    return round(float(np.sum(er) * (sp**2) / 1e6), 4), round(float(np.sum(ac) * (sp**2) / 1e6), 4)


def export_shapefile_zip(edges, metadata, h5_path):
    """
    Same detection input/output as before, only the export step is
    changed: it now also returns the in-memory GeoDataFrame alongside
    the zip bytes, so the map preview doesn't need to re-extract the
    zip and re-read the shapefile from disk.
    """
    cnts, _ = cv2.findContours(edges.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    main_cnt = max(cnts, key=lambda c: cv2.arcLength(c, False))

    with h5py.File(h5_path, 'r') as f:
        target = find_nisar_main_path(f)
        if not target:
            raise ValueError("Could not locate NISAR grid path in HDF5 file.")
        grid_path = "/".join(target.split("/")[:-1])
        x_coords = f[f"{grid_path}/xCoordinates"][()]
        y_coords = f[f"{grid_path}/yCoordinates"][()]
        proj_val = f[f"{grid_path}/projection"][()]

    src_crs = CRS.from_epsg(int(proj_val))
    transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)

    max_col = len(x_coords) - 1
    max_row = len(y_coords) - 1

    geo_points = []
    for i in range(len(main_cnt)):
        px_x = float(main_cnt[i, 0, 0])
        px_y = float(main_cnt[i, 0, 1])

        col = int(round(px_x * 6.0))
        row = int(round(px_y * 6.0))
        col = min(max(0, col), max_col)
        row = min(max(0, row), max_row)

        utm_x = x_coords[col]
        utm_y = y_coords[row]

        lon, lat = transformer.transform(utm_x, utm_y)
        geo_points.append((lon, lat))

    if len(geo_points) < 2:
        return None, None
    line_geom = LineString(geo_points)

    gdf = gpd.GeoDataFrame(geometry=[line_geom], crs="EPSG:4326")
    gdf['Id'] = 0
    gdf['Feature'] = ["NISAR Core Shoreline Layer Vector"]

    with tempfile.TemporaryDirectory() as tmp_dir:
        base_name = "shoreline"
        shp_path = os.path.join(tmp_dir, f"{base_name}.shp")
        gdf.to_file(shp_path, driver="ESRI Shapefile")

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as z:
            for ext in ['.shp', '.shx', '.dbf', '.prj']:
                f_path = os.path.join(tmp_dir, f"{base_name}{ext}")
                if os.path.exists(f_path):
                    z.write(f_path, arcname=f"{base_name}{ext}")

        zip_buffer.seek(0)
        return zip_buffer.getvalue(), gdf


def load_india_boundary(shp_path):
    """
    Parses the India reference shapefile/zip once. The app caches the
    returned GeoDataFrame in session_state so this isn't re-run inside
    every expander/loop iteration.
    """
    import glob
    try:
        if shp_path.endswith('.zip'):
            with tempfile.TemporaryDirectory() as tmp_ind:
                with zipfile.ZipFile(shp_path, 'r') as z_ind:
                    z_ind.extractall(tmp_ind)
                shps = glob.glob(os.path.join(tmp_ind, "**", "*.shp"), recursive=True)
                if shps:
                    return gpd.read_file(shps[0]).to_crs("EPSG:4326")
        else:
            return gpd.read_file(shp_path).to_crs("EPSG:4326")
    except Exception:
        return None
    return None