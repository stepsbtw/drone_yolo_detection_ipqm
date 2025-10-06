import numpy as np

class Camera:
    # Drone EVO 2 Dual V2
    def __init__(self, sensor_width_mm=6.4, sensor_height_mm=4.8,
                 focal_35mm_mm=25.6, image_width_px=1920, image_height_px=1080):

        self.sensor_width_mm = sensor_width_mm
        self.sensor_height_mm = sensor_height_mm
        self.focal_35mm_mm = focal_35mm_mm
        self.image_width_px = image_width_px
        self.image_height_px = image_height_px

        # real focal length calculation
        self.diag_sensor_mm = np.sqrt(sensor_width_mm**2 + sensor_height_mm**2)
        self.diag_35mm_mm = 43.27  # diagonal of 35mm sensor
        self.crop_factor = self.diag_35mm_mm / self.diag_sensor_mm
        self.focal_length_mm = self.focal_35mm_mm / self.crop_factor

        self.pixel_size_x_mm = sensor_width_mm / image_width_px
        self.pixel_size_y_mm = sensor_height_mm / image_height_px


    def estimate_distance(self, pixel_height, real_height_m=1.7):

        focal_length_m = self.focal_length_mm / 1000
        pixel_size_y_m = self.pixel_size_y_mm / 1000
        
        distance_m = (real_height_m * focal_length_m) / (pixel_height * pixel_size_y_m)
        return distance_m