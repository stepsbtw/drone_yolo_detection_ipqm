"""
exemplo de uso do metodo dcm (distance calculation method)
para estimativa precisa de distancia com coordenadas geograficas
"""

from drone_people_detector import DronePeopleDetector
from drone_people_detector.core.estimation import Camera

# configuracao 1: metodo pinhole simples (padrao)
# apenas distancia estimada
camera_simple = Camera(
    sensor_width_mm=6.4,
    sensor_height_mm=4.8,
    focal_35mm_mm=25.6,
    image_width_px=1920,
    image_height_px=1080
)

# configuracao 2: metodo dcm (completo)
# distancia + coordenadas geograficas + bearing
camera_dcm = Camera(
    sensor_width_mm=6.4,
    sensor_height_mm=4.8,
    focal_35mm_mm=25.6,
    image_width_px=1920,
    image_height_px=1080,
    hfov=62.2,  # campo de visao horizontal em graus
    bearing=45,  # direcao que camera aponta (norte=0, leste=90)
    lat=-23.5505,  # latitude da camera
    lon=-46.6333,  # longitude da camera
    x=326000,  # posicao utm x (opcional)
    y=7395000   # posicao utm y (opcional)
)

# detector
detector = DronePeopleDetector(
    model_path='models/people/yolo11n.pt',
    confidence_threshold=0.5,
    enable_weapon_detection=True,
    enable_tracking=True
)

# processar video
# metodo pinhole simples (retorna apenas distancia)
annotated_frame, tracks = detector.process_frame_with_tracking(frame, camera=camera_simple)
for track in tracks:
    print(f"Track {track.id}: distance={track.distance:.1f}m")

# metodo dcm (retorna distancia + coordenadas)
# automaticamente detectado se camera tiver hfov configurado
annotated_frame, tracks = detector.process_frame_with_tracking(frame, camera=camera_dcm)
for track in tracks:
    # para usar coordenadas, precisa chamar dcm diretamente:
    if track.bbox:
        x, y, lat, lon, bearing, distance = camera_dcm.estimate_distance_dcm(track.bbox)
        print(f"Track {track.id}:")
        print(f"  Distance: {distance:.1f}m")
        print(f"  Bearing: {bearing:.1f}°")
        if lat and lon:
            print(f"  GPS: {lat:.6f}, {lon:.6f}")
