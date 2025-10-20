import numpy as np

class Camera:
    """camera para estimativa de distancia (drone evo 2 dual v2)"""
    
    def __init__(self, sensor_width_mm=6.4, sensor_height_mm=4.8,
                 focal_35mm_mm=25.6, image_width_px=1920, image_height_px=1080,
                 hfov=None, bearing=0, x=0, y=0, lat=None, lon=None):
        """
        hfov: campo de visao horizontal em graus (opcional, para dcm)
        bearing: direcao da camera em graus (opcional, para dcm)
        x, y: posicao utm da camera (opcional, para dcm)
        lat, lon: coordenadas geograficas (opcional, para dcm)
        """
        self.sensor_width_mm = sensor_width_mm
        self.sensor_height_mm = sensor_height_mm
        self.focal_35mm_mm = focal_35mm_mm
        self.image_width_px = image_width_px
        self.image_height_px = image_height_px
        
        self.sensor_width_resolution = image_width_px
        self.sensor_height_resolution = image_height_px

        # calculo do focal length real
        self.diag_sensor_mm = np.sqrt(sensor_width_mm**2 + sensor_height_mm**2)
        self.diag_35mm_mm = 43.27
        self.crop_factor = self.diag_35mm_mm / self.diag_sensor_mm
        self.focal_length_mm = self.focal_35mm_mm / self.crop_factor
        self.focal_length_px = (self.focal_length_mm / sensor_width_mm) * image_width_px

        self.pixel_size_x_mm = sensor_width_mm / image_width_px
        self.pixel_size_y_mm = sensor_height_mm / image_height_px
        
        # parametros para dcm
        self.hfov = hfov
        self.bearing = bearing
        self.x = x
        self.y = y
        self.lat = lat
        self.lon = lon

    def estimate_distance(self, pixel_height, real_height_m=1.7):
        """metodo pinhole simples para estimativa de distancia"""
        focal_length_m = self.focal_length_mm / 1000
        pixel_size_y_m = self.pixel_size_y_mm / 1000
        
        distance_m = (real_height_m * focal_length_m) / (pixel_height * pixel_size_y_m)
        return distance_m
    
    def estimate_distance_dcm(self, detected_bbox, real_height_m=1.7):
        """
        metodo dcm (distance calculation method)
        considera posicao horizontal no frame, bearing e hfov
        retorna: (x, y, lat, lon, bearing, distance)
        """
        if self.hfov is None:
            raise ValueError("hfov necessario para dcm")
        
        try:
            from drone_people_detector.core.monocular_vision_submodule import MonocularVision
            return MonocularVision.monocular_vision_detection_method_2(
                self, real_height_m, detected_bbox
            )
        except ImportError:
            # fallback para metodo simples se modulo nao disponivel
            pixel_height = detected_bbox[3]
            distance = self.estimate_distance(pixel_height, real_height_m)
            return None, None, None, None, self.bearing, distance