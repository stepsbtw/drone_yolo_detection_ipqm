import time
from imutils.video import VideoStream
import math
import numpy as np
from datetime import datetime, timezone
import logging
from drone_people_detector.core.converter import Converter

logging.basicConfig(level=logging.INFO, format='%(message)s', )

class Camera:
    """
    Camera para estimativa de distancia - Autel EVO II Dual V2
    
    Especificacoes oficiais:
    - Sensor: 1/2" CMOS, 48MP (8000x6000 pixels)
    - Focal Length (EFL): 25.6mm
    - Aperture: f/2.8 - f/11
    - HFOV: 79°
    - Zoom: 1-8x (Max 4x lossless)
    - Focus Distance: 0.5m to infinity
    - Max Flight Altitude: 7000m MSL
    """
    
    # Constantes das especificacoes oficiais do Autel EVO II Dual V2
    NATIVE_WIDTH_PX = 8000  # pixels (48MP)
    NATIVE_HEIGHT_PX = 6000  # pixels (4:3 aspect ratio)
    SENSOR_WIDTH_MM = 6.4  # mm (1/2" CMOS sensor standard)
    SENSOR_HEIGHT_MM = 4.8  # mm (calculated from 4:3 ratio)
    FOCAL_LENGTH_EFL_MM = 25.6  # mm (Equivalent Focal Length)
    HFOV_DEG = 79.0  # degrees
    MIN_FOCUS_DISTANCE_M = 0.5  # meters
    MAX_ALTITUDE_M = 7000.0  # meters MSL
    APERTURE_MIN = 2.8  # f-stop
    APERTURE_MAX = 11.0  # f-stop
    ZOOM_MIN = 1.0
    ZOOM_MAX = 8.0
    ZOOM_LOSSLESS_MAX = 4.0  # Maximum lossless zoom

    def __init__(self, site=None, sensor_width_mm=None, sensor_height_mm=None,
                 focal_35mm_mm=None, image_width_px=1920, image_height_px=1080,
                 bearing=0, lat=None, lon=None, zoom=1.0):
        """
        Inicializa camera com especificacoes do Autel EVO II Dual V2.
        
        Parametros:
            site: dict com configuracao PTZ (opcional, para cameras PTZ)
            sensor_width_mm: largura do sensor em mm (default: 6.4mm para 1/2" CMOS)
            sensor_height_mm: altura do sensor em mm (default: 4.8mm)
            focal_35mm_mm: focal length equivalente (default: 25.6mm)
            image_width_px: largura da imagem processada em pixels (default: 1920)
            image_height_px: altura da imagem processada em pixels (default: 1080)
            bearing: direcao da camera em graus (opcional, para dcm)
            lat, lon: coordenadas geograficas (opcional, para dcm)
            zoom: fator de zoom atual (1.0-8.0, default: 1.0)
        """
        self.focused_track = None
        
        # Se site config fornecido, inicializar como camera PTZ
        if site is not None:
            self.load_config(site)
            self.set_ptz(0, 0, 0)
            if self.address is not None:
                self.video_stream = VideoStream(self.address, frame_rate=self.frame_rate,
                                            resolution=(self.sensor_width_resolution, self.sensor_height_resolution))
        else:
            # Inicializar como camera de drone (Autel EVO II Dual V2)
            self.address = None
            self.video_stream = None
            
            # Usar especificacoes oficiais se nao fornecidas
            self.sensor_width_mm = sensor_width_mm or self.SENSOR_WIDTH_MM
            self.sensor_height_mm = sensor_height_mm or self.SENSOR_HEIGHT_MM
            self.focal_35mm_mm = focal_35mm_mm or self.FOCAL_LENGTH_EFL_MM
            self.image_width_px = image_width_px
            self.image_height_px = image_height_px
            
            self.sensor_width_resolution = image_width_px
            self.sensor_height_resolution = image_height_px

            # Calculo do focal length real
            self.diag_sensor_mm = np.sqrt(self.sensor_width_mm**2 + self.sensor_height_mm**2)
            self.diag_35mm_mm = 43.27
            self.crop_factor = self.diag_35mm_mm / self.diag_sensor_mm
            self.focal_length_mm = self.focal_35mm_mm / self.crop_factor
            self.focal_length_px = (self.focal_length_mm / self.sensor_width_mm) * image_width_px

            self.pixel_size_x_mm = self.sensor_width_mm / image_width_px
            self.pixel_size_y_mm = self.sensor_height_mm / image_height_px
            
            # Campo de visao (FOV)
            self.hfov = self.HFOV_DEG  # Usar specs oficiais
            
            # Calcular VFOV baseado no aspect ratio da imagem
            aspect_ratio = image_height_px / image_width_px
            self.vfov = 2 * np.degrees(np.arctan(np.tan(np.radians(self.hfov / 2)) * aspect_ratio))
            
            # Calcular HFOV teorico para validacao
            self.hfov_calculated = 2 * np.degrees(np.arctan(self.sensor_width_mm / (2 * self.focal_length_mm)))
            
            # Zoom (afeta FOV efetivo)
            self.zoom_multiplier = np.clip(zoom, self.ZOOM_MIN, self.ZOOM_MAX)
            self.effective_hfov = self.hfov / self.zoom_multiplier
            self.effective_vfov = self.vfov / self.zoom_multiplier
            
            # Parametros para DCM (Distance Calculation Method)
            self.bearing = bearing
            self.lat = lat
            self.lon = lon
            
            # Calcular focal length em pixels baseado no HFOV efetivo
            self.focal_length_px_from_hfov = (image_width_px / 2) / np.tan(np.radians(self.effective_hfov / 2))
            
            # Log das especificacoes para debug
            self._log_camera_specs()
    
    def _log_camera_specs(self):
        """Exibe as especificacoes da camera para validacao"""
        print("\n" + "="*70)
        print("AUTEL EVO II DUAL V2 - Camera Specifications")
        print("="*70)
        print(f"Sensor:")
        print(f"  Type: 1/2\" CMOS, 48MP")
        print(f"  Native Resolution: {self.NATIVE_WIDTH_PX}x{self.NATIVE_HEIGHT_PX} pixels")
        print(f"  Physical Size: {self.sensor_width_mm:.2f}mm x {self.sensor_height_mm:.2f}mm")
        print(f"  Pixel Size: {self.pixel_size_x_mm*1000:.2f}µm x {self.pixel_size_y_mm*1000:.2f}µm")
        print(f"\nLens:")
        print(f"  Focal Length (EFL): {self.focal_35mm_mm:.1f}mm")
        print(f"  Focal Length (Real): {self.focal_length_mm:.2f}mm")
        print(f"  Aperture Range: f/{self.APERTURE_MIN} - f/{self.APERTURE_MAX}")
        print(f"  Focus Range: {self.MIN_FOCUS_DISTANCE_M}m to infinity")
        print(f"\nField of View:")
        print(f"  HFOV (Spec): {self.HFOV_DEG}°")
        print(f"  HFOV (Calculated): {self.hfov_calculated:.2f}°")
        print(f"  VFOV: {self.vfov:.2f}°")
        print(f"\nZoom:")
        print(f"  Current Zoom: {self.zoom_multiplier}x")
        print(f"  Effective HFOV: {self.effective_hfov:.2f}°")
        print(f"  Effective VFOV: {self.effective_vfov:.2f}°")
        print(f"  Zoom Range: {self.ZOOM_MIN}x - {self.ZOOM_MAX}x (Lossless up to {self.ZOOM_LOSSLESS_MAX}x)")
        print(f"\nImage Processing:")
        print(f"  Working Resolution: {self.image_width_px}x{self.image_height_px} pixels")
        print(f"  Focal Length (pixels): {self.focal_length_px:.2f}px")
        print(f"  Focal Length from HFOV (pixels): {self.focal_length_px_from_hfov:.2f}px")
        print(f"  Crop Factor: {self.crop_factor:.2f}x")
        print(f"\nOperational Limits:")
        print(f"  Max Flight Altitude: {self.MAX_ALTITUDE_M}m MSL")
        print(f"  Min Detection Distance: {self.MIN_FOCUS_DISTANCE_M}m")
        print("="*70 + "\n")
    
    def set_zoom(self, zoom_factor):
        """
        Atualiza o fator de zoom e recalcula FOV efetivo
        
        Args:
            zoom_factor: Fator de zoom (1.0-8.0)
        """
        self.zoom_multiplier = np.clip(zoom_factor, self.ZOOM_MIN, self.ZOOM_MAX)
        self.effective_hfov = self.hfov / self.zoom_multiplier
        self.effective_vfov = self.vfov / self.zoom_multiplier
        self.focal_length_px_from_hfov = (self.image_width_px / 2) / np.tan(np.radians(self.effective_hfov / 2))
        
        if self.zoom_multiplier > self.ZOOM_LOSSLESS_MAX:
            print(f"⚠️  Warning: Zoom {self.zoom_multiplier}x excede zoom lossless ({self.ZOOM_LOSSLESS_MAX}x)")
        
        print(f"Zoom atualizado: {self.zoom_multiplier}x | Effective HFOV: {self.effective_hfov:.2f}°")
    
    def estimate_distance(self, pixel_height, real_height_m=1.7):
        """
        Metodo pinhole simples para estimativa de distancia
        
        Args:
            pixel_height: Altura do objeto detectado em pixels
            real_height_m: Altura real do objeto em metros (default: 1.7m para pessoa)
            
        Returns:
            distance_m: Distancia estimada em metros
        """
        if pixel_height <= 0:
            raise ValueError(f"pixel_height deve ser positivo, recebido: {pixel_height}")
        
        focal_length_m = self.focal_length_mm / 1000
        pixel_size_y_m = self.pixel_size_y_mm / 1000
        
        distance_m = (real_height_m * focal_length_m) / (pixel_height * pixel_size_y_m)
        
        # Validacao baseada nas especificacoes
        if distance_m < self.MIN_FOCUS_DISTANCE_M:
            print(f"⚠️  Warning: Distancia estimada ({distance_m:.2f}m) abaixo do minimo de foco ({self.MIN_FOCUS_DISTANCE_M}m)")
        
        return distance_m
    
    def estimate_distance_with_hfov(self, pixel_height, real_height_m=1.7):
        """
        Metodo alternativo usando focal length calculado a partir do HFOV
        Recomendado para maior precisao com as specs oficiais
        
        Args:
            pixel_height: Altura do objeto detectado em pixels
            real_height_m: Altura real do objeto em metros (default: 1.7m para pessoa)
            
        Returns:
            distance_m: Distancia estimada em metros
        """
        if pixel_height <= 0:
            raise ValueError(f"pixel_height deve ser positivo, recebido: {pixel_height}")
        
        # Usar focal length calculado a partir do HFOV efetivo
        distance_m = (real_height_m * self.focal_length_px_from_hfov) / pixel_height
        
        # Validacao baseada nas especificacoes
        if distance_m < self.MIN_FOCUS_DISTANCE_M:
            print(f"⚠️  Warning: Distancia estimada ({distance_m:.2f}m) abaixo do minimo de foco ({self.MIN_FOCUS_DISTANCE_M}m)")
        
        return distance_m
    

    

    

    
    def estimate_distance_dcm(self, detected_bbox, real_height_m=1.7):
        """
        Metodo DCM (Distance Calculation Method)
        Considera posicao horizontal no frame, bearing e HFOV
        
        Args:
            detected_bbox: Bounding box da deteccao [x, y, width, height]
            real_height_m: Altura real do objeto em metros (default: 1.7m)
            
        Returns:
            tuple: (x, y, lat, lon, bearing, distance)
                - x, y: Posicao UTM estimada
                - lat, lon: Coordenadas geograficas estimadas
                - bearing: Bearing ajustado para o objeto
                - distance: Distancia estimada em metros
        """
        if self.hfov is None:
            raise ValueError("HFOV necessario para DCM. Use hfov=79.0 para Autel EVO II Dual V2")
        
        try:
            from drone_people_detector.core.monocular_vision_submodule import MonocularVision
            return MonocularVision.monocular_vision_detection_method_2(
                self, real_height_m, detected_bbox
            )
        except ImportError as e:
            print(f"⚠️  Warning: Modulo MonocularVision nao disponivel: {e}")
            print("   Usando fallback para metodo simples...")
            # Fallback para metodo simples se modulo nao disponivel
            pixel_height = detected_bbox[3]
            distance = self.estimate_distance_with_hfov(pixel_height, real_height_m)
            return None, None, None, None, self.bearing, distance
    
    def get_specs_dict(self):
        """
        Retorna um dicionario com todas as especificacoes da camera
        Util para logging e debugging
        
        Returns:
            dict: Especificacoes completas da camera
        """
        return {
            'model': 'Autel EVO II Dual V2',
            'sensor': {
                'type': '1/2" CMOS',
                'megapixels': 48,
                'native_resolution': f"{self.NATIVE_WIDTH_PX}x{self.NATIVE_HEIGHT_PX}",
                'width_mm': self.sensor_width_mm,
                'height_mm': self.sensor_height_mm,
                'pixel_size_um': round(self.pixel_size_x_mm * 1000, 2)
            },
            'lens': {
                'focal_length_efl_mm': self.focal_35mm_mm,
                'focal_length_real_mm': round(self.focal_length_mm, 2),
                'aperture_range': f"f/{self.APERTURE_MIN}-f/{self.APERTURE_MAX}",
                'focus_range_m': f"{self.MIN_FOCUS_DISTANCE_M} to infinity"
            },
            'fov': {
                'hfov_deg': self.hfov,
                'vfov_deg': round(self.vfov, 2),
                'effective_hfov_deg': round(self.effective_hfov, 2),
                'effective_vfov_deg': round(self.effective_vfov, 2)
            },
            'zoom': {
                'current': self.zoom_multiplier,
                'range': f"{self.ZOOM_MIN}x-{self.ZOOM_MAX}x",
                'lossless_max': f"{self.ZOOM_LOSSLESS_MAX}x"
            },
            'processing': {
                'resolution': f"{self.image_width_px}x{self.image_height_px}",
                'focal_length_px': round(self.focal_length_px, 2),
                'focal_length_px_from_hfov': round(self.focal_length_px_from_hfov, 2),
                'crop_factor': round(self.crop_factor, 2)
            },
            'operational': {
                'max_altitude_m': self.MAX_ALTITUDE_M,
                'min_focus_distance_m': self.MIN_FOCUS_DISTANCE_M
            },
            'position': {
                'bearing_deg': self.bearing,
                'lat': self.lat,
                'lon': self.lon
            }
        }

    def load_config(self, camera_data):
        self.id = camera_data['id']
        self.address = camera_data['address']
        self.lat = camera_data['latitude']
        self.lon = camera_data['longitude']
        self.x, self.y = Converter.geo_to_xy(self.lat, self.lon)
        self.ref_elevation= camera_data['reference_elevation']
        self.ref_azimuth = camera_data['reference_azimuth']
        self.installation_height = camera_data['installation_height']
        self.surveillance_radius = camera_data ['surveillance_radius']
        self.focus_frame_view = camera_data ['focus_frame_view']
        self.sensor_height_lens = camera_data ['sensor_height_lens']
        self.sensor_width_lens = camera_data ['sensor_width_lens']
        self.zoom_multiplier_min = camera_data['zoom_multiplier_min']
        self.zoom_multiplier_max = camera_data['zoom_multiplier_max']
        self.zoom_lens_min = camera_data['zoom_lens_min']
        self.zoom_lens_max = camera_data['zoom_lens_max']
        self.hfov_min = camera_data['hfov_min']
        self.hfov_max = camera_data['hfov_max']
        self.sensor_height_resolution = camera_data['sensor_height_resolution']
        self.sensor_width_resolution = camera_data['sensor_width_resolution']
        self.frame_rate = camera_data['frame_rate']
        self.bearing = 0
        self.elevation = 0
        self.zoom_multiplier = self.zoom_multiplier_min
        self.pan = 0
        self.tilt = 0
        self.zoom = 0
        self.manual_offset_pan = 0
        self.manual_offset_tilt = 0
        self.manual_offset_zoom = 0
        self.focused_track = None
        self.resolution_ratio = self.sensor_width_resolution / self.sensor_height_resolution
        self.timestamp = datetime.now()
        self.timestamp_ptz = datetime.now()
        self.current_frame_rate = self.frame_rate
        self.interval_measured = 0
        
        if self.address is not None:
            logging.info('Address: ' + self.address)
        logging.info('Geo Position: ' + str(self.lat) + '  ' + str(self.lon))
        logging.info('Reference elevation (degrees): ' + str(self.ref_elevation))
        logging.info('Reference azimuth (degrees): ' + str(self.ref_azimuth))
        logging.info('Initial Bearing (degrees): ' + str(self.bearing))
        logging.info('Initial zoom (multiplier): ' + str(self.zoom_multiplier))
        logging.info('Installation height (m): ' + str(self.installation_height))
        logging.info('Surveillance Radius (m): ' + str(self.surveillance_radius))
        logging.info('Focus Frame View (px): ' + str(self.focus_frame_view))
        logging.info('Sensor Width (mm): ' + str(self.sensor_width_lens))
        logging.info('Sensor Height (mm): ' + str(self.sensor_height_lens))
        logging.info('Zoom Max (multiplier): ' + str(self.zoom_multiplier_max))
        logging.info('Zoom Min (multiplier): ' + str(self.zoom_multiplier_min))
        logging.info('Zoom Lens Max (mm): ' + str(self.zoom_lens_max))
        logging.info('Zoom Lens Min (mm): ' + str(self.zoom_lens_min))
        logging.info('HFOV Min (degrees): ' + str(self.hfov_max))
        logging.info('HFOV Max (degrees): ' + str(self.hfov_min))
        logging.info('Expected Frame Rate (FPS): ' + str(self.frame_rate))
        logging.info('Sensor Resolution - Ratio: (' + str(self.sensor_width_resolution) + ',' + str(self.sensor_height_resolution) + ') - (' + str(self.resolution_ratio)+ ')') 
       
    def next_frame(self):
        (ok, frame) = self.video_stream.stream.stream.read()
        previous_timestamp = self.timestamp
        self.timestamp = datetime.now()
        self.interval_measured = ((self.timestamp - previous_timestamp).microseconds / 1e+6)
        self.current_frame_rate = 1 / self.interval_measured

        return frame
    
    def tracking(self, track):
        x, y = Converter.geo_to_xy(track.lat, track.lon)
        track.utm.position = (x, y)
        track.utm.timestamp = datetime.now(timezone.utc)
        interval = (track.utm.timestamp - self.timestamp_ptz).seconds
               
        vx, vy = Converter.polar_to_xy(0,0,track.course,track.speed)
        track.utm.linear_estimation(vx, vy, interval)
        lat, lon = Converter.xy_to_geo(track.utm.position[0],track.utm.position[1])
        bearing, distance = Converter.geo_to_polar(self.lat,self.lon,lat,lon)
        
        self.set_to_track_position(bearing,distance)
    

    def estimate_tilt(self, distance):
        theta = math.atan2(self.installation_height, distance)
        theta = math.degrees(theta)
        theta += self.ref_elevation
        if theta > 360:
            theta = theta - 360
        t = -((theta + 90) / 180 * 2 - 1)
        self.elevation = t * 90
        return t

    def estimate_pan(self, bearing):
        self.bearing = bearing
        bearing = bearing + (360 - self.ref_azimuth)
        
        if bearing >= 360:
            bearing = bearing - 360
        
        if bearing <= 180:
            p = bearing / 180
        else:
            p = ((360 - bearing) / 180) * -1
        
        return p

    def estimate_zoom_factor_survaillance_radius(self,distance):
        if (self.installation_height + distance) > self.surveillance_radius:
            self.zoom_multiplier = self.zoom_multiplier_max
        else:
            self.zoom_multiplier = ((self.installation_height + distance) / self.surveillance_radius) * self.zoom_multiplier_max
        z  = (self.zoom_multiplier - self.zoom_multiplier_min) / (self.zoom_multiplier_max - self.zoom_multiplier_min)
        return z

    def estimate_zoom_factor_by_focal_estimation(self, distance):
        if distance <= 0:
            return 0

        mm_per_pixel = 25.4 / 96  # baseado em DPI
        scene_width = self.focus_frame_view * mm_per_pixel

        theta = 2 * math.atan(scene_width / (2 * distance))
        required_focal_length_mm = self.sensor_width_lens / (2 * math.tan(theta / 2))
        z = required_focal_length_mm / self.zoom_lens_max

        return z
    

    
    def set_to_track_position(self, bearing, distance):
        p = 0
        t = 0
        z = 0
        
        z = self.estimate_zoom_factor_by_focal_estimation(distance)
        t = self.estimate_tilt(distance)
        p = self.estimate_pan(bearing)
        
        self.set_ptz(p,t,z)

    def convert_ptz_to_polar(self, pan, tilt, zoom):
        if pan >= 0:
            bearing = pan * 180
        else:
            bearing = 360 - ((pan*-1) * 180)
        bearing = bearing + self.ref_azimuth
        if bearing > 360:
            bearing = bearing - 360
        self.bearing = bearing
        
        self.elevation = tilt * 90
        self.elevation += self.ref_elevation
   
        self.zoom_multiplier = (self.zoom_multiplier_max - self.zoom_multiplier_min) * zoom
        self.set_ptz(pan,tilt,zoom)
        
    def set_ptz(self, pan, tilt, zoom):
        pan = pan + self.manual_offset_pan
        tilt = tilt + self.manual_offset_tilt
        zoom = zoom + self.manual_offset_zoom

        if pan > 1:
            pan = 1
        if pan < -1:
            pan = -1
        
        if tilt > 1:
            tilt = 1
        if tilt < -1:
            tilt = -1
        
        if zoom > 1:
            zoom = 1
        if zoom < 0:
            zoom = 0
        
        self.pan = pan
        self.tilt = tilt
        self.zoom = zoom

        self.calculate_new_focal_length(self.zoom)
    
    def calculate_new_focal_length(self,zoom):
        self.focal_length_mm = self.zoom_lens_min + ((self.zoom_lens_max - self.zoom_lens_min) * zoom)
        self.hfov = self.hfov_min + ((self.hfov_max - self.hfov_min) * (1-zoom))
        self.focal_length_px = (self.focal_length_mm * self.sensor_height_resolution) / self.sensor_height_lens
