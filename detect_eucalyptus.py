import argparse
import json
import os
from ultralytics import YOLO
from pathlib import Path
import rasterio
import numpy as np
import geopandas as gpd
from shapely.geometry import box

def calculate_gini(areas):
    # Baseado na fórmula de Gini para desigualdade
    sorted_areas = np.sort(areas)
    n = len(areas)
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * sorted_areas)) / (n * np.sum(sorted_areas))

def calculate_pv50(areas):
    # Ordena as áreas e pega as 50% menores
    sorted_areas = np.sort(areas)
    half_n = len(sorted_areas) // 2
    sum_smallest_50 = np.sum(sorted_areas[:half_n])
    total_sum = np.sum(sorted_areas)
    return (sum_smallest_50 / total_sum) * 100

def main():
    parser = argparse.ArgumentParser(description="Detect eucalyptus sprouts in images using YOLO model.")
    parser.add_argument('--source', type=str, required=True, help='Path to image file.')
    parser.add_argument('--output', type=str, default='results', help='Directory to save the output files results')
    parser.add_argument('--model', type=str, default='./runs/detect/eucalipto/v5/weights/best.pt', help='Path to the YOLO model file (.pt)')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold for detection')
    parser.add_argument('--gsd', type=float, default=0.05, help='Ground Sampling Distance (GSD) in meters per pixel')
    
    args = parser.parse_args()

    # Checar se o caminho da imagem é válido e se o formato é suportado
    if not os.path.isfile(args.source):
        print(f"Source path {args.source} is not a file.")
        return
    elif not args.source.lower().endswith(('.tif', '.tiff')):
        print(f"Source file {args.source} is not a supported image format.")
        return

    # Checar se o modelo existe e se os parâmetros são válidos
    if not os.path.isfile(args.model):
        print(f"Model file {args.model} does not exist.")
        return
    if not (0 <= args.conf <= 1):
        print("Confidence threshold must be between 0 and 1.")
        return
    if not (args.gsd > 0):
        print("GSD must be a positive number.")
        return
    
    # Carregar o modelo YOLO
    model = YOLO(args.model)
    print("Modelo carregado com sucesso.")

    with rasterio.open(args.source) as src:
        # Ler a imagem (ajustando para o formato que o YOLO entende: HWC)
        img = src.read([1, 2, 3]) # Lê as bandas 1, 2 e 3 (RGB)
        img = np.moveaxis(img, 0, -1) # De (C, H, W) para (H, W, C)
        
        # Guardar a transformação (matriz que converte pixel -> coordenada real)
        transform = src.transform 
    print("Imagem carregada com sucesso. Iniciando a detecção...")

    # Fazer a detecção
    results = model.predict(img, conf=args.conf, save=False)
    boxes = results[0].boxes.xyxy.cpu().numpy() # [x1, y1, x2, y2] em pixels
    classes = results[0].boxes.cls.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()

    print(f"Detecções realizadas: {len(boxes)}. Convertendo para vetores geográficos...")

    # Converter Bounding Boxes de Pixels para Vetores Geográficos
    polygons = []
    metadata = []

    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = b
        
        # Converte as coordenadas de pixel para as coordenadas do TIFF
        # Se o TIFF não tiver coordenadas reais, ele usará a escala de pixels
        lon_min, lat_max = transform * (x1, y1)
        lon_max, lat_min = transform * (x2, y2)
        
        # Criar um polígono (retângulo)
        poly = box(lon_min, lat_min, lon_max, lat_max)
        polygons.append(poly)
        
        metadata.append({
            'class': int(classes[i]),
            'confidence': float(confidences[i])
        })

    # Criar um GeoDataFrame
    gdf = gpd.GeoDataFrame(metadata, geometry=polygons, crs=src.crs)

    print(f"Vetorização concluída. Total de vetores criados: {len(gdf)}. Calculando áreas e métricas...")

    # Conversão para utm para calcular áreas em metros quadrados
    estimated_utm = gdf.estimate_utm_crs()
    gdf = gdf.to_crs(estimated_utm)

    # Calculando a área
    gdf['area'] = gdf.geometry.area

    # Filtrar por área mínima (1 m²) para evitar detecções muito pequenas
    gdf = gdf[gdf.geometry.area > 1]

    # Cacular a quantidade total de hectares da imagem, desconsiderando no data
    total_area_ha = (np.sum(np.where((img[:,:,0] + img[:,:,1] + img[:,:,2])>0,1,0)) * (args.gsd ** 2)) / 10000

    # Calculo da métricas para exportar em um json
    total_detections = len(gdf)
    total_detections_per_ha = total_detections / total_area_ha if total_area_ha > 0 else 0

    # Métrica de homogenidade da plantas detectadas (exemplo: desvio padrão da área das detecções)
    desvio_padrao = gdf['area'].std() if len(gdf) > 1 else 0

    metrics = {
        'total_detections': total_detections,
        'total_area_ha': total_area_ha,
        'detections_per_ha': total_detections_per_ha,
        'desvio_padrao': desvio_padrao,
        'gini': calculate_gini(gdf['area']),
        'pv50': calculate_pv50(gdf['area'])
    }

    print("Métricas calculadas. Salvando resultados...")
    
    # Salvar os vetores (em geojson) e as métricas
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    gdf = gdf.to_crs(4326) # Converter para WGS84 para salvar em GeoJSON
    gdf.to_file(output_dir / "detections.geojson", driver="GeoJSON")
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Processo concluído! Vetores salvos em '{output_dir / 'detections.geojson'}' e métricas em '{output_dir / 'metrics.json'}'.")

if __name__ == "__main__":
    main()