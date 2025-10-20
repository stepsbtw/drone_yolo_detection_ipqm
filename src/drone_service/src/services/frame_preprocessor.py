"""
preprocessador de frames para otimizacao de tempo real
reduz resolucao e implementa frame skipping
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FramePreprocessor:
    """preprocessa frames para deteccao em tempo real"""
    
    def __init__(
        self,
        target_width: int = 1280,
        target_height: int = 720,
        frame_skip: int = 1,
        maintain_aspect_ratio: bool = True
    ):
        """
        target_width: largura alvo (None = sem resize)
        target_height: altura alvo (None = sem resize)
        frame_skip: processar 1 a cada N frames (1 = todos)
        maintain_aspect_ratio: manter proporcao original
        """
        self.target_width = target_width
        self.target_height = target_height
        self.frame_skip = frame_skip
        self.maintain_aspect_ratio = maintain_aspect_ratio
        
        self.frame_count = 0
        self.last_processed_frame = None
        self.last_processed_data = None
        
        # stats
        self.frames_skipped = 0
        self.frames_processed = 0
        self.resize_enabled = target_width is not None or target_height is not None
        
        logger.info(f"preprocessador inicializado:")
        logger.info(f"  target: {target_width}x{target_height}")
        logger.info(f"  frame skip: 1/{frame_skip}")
        logger.info(f"  resize: {self.resize_enabled}")
    
    def should_process_frame(self) -> bool:
        """verifica se frame atual deve ser processado"""
        self.frame_count += 1
        should_process = (self.frame_count % self.frame_skip) == 0
        
        if should_process:
            self.frames_processed += 1
        else:
            self.frames_skipped += 1
        
        return should_process
    
    def resize_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        reduz resolucao do frame
        retorna: (frame_reduzido, scale_factor)
        """
        if not self.resize_enabled:
            return frame, 1.0
        
        original_height, original_width = frame.shape[:2]
        
        # calcula dimensoes alvo
        if self.maintain_aspect_ratio:
            # reduz mantendo aspect ratio
            if self.target_height is not None:
                scale = self.target_height / original_height
            else:
                scale = self.target_width / original_width
            
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
        else:
            new_width = self.target_width or original_width
            new_height = self.target_height or original_height
            scale = new_height / original_height
        
        # verifica se precisa reduzir
        if new_width >= original_width and new_height >= original_height:
            return frame, 1.0
        
        # resize com interpolacao linear (mais rapida)
        resized = cv2.resize(
            frame,
            (new_width, new_height),
            interpolation=cv2.INTER_LINEAR
        )
        
        return resized, scale
    
    def preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, bool, float]:
        """
        preprocessa frame
        retorna: (frame_processado, foi_processado, scale_factor)
        """
        should_process = self.should_process_frame()
        
        if should_process:
            # processa novo frame
            processed_frame, scale = self.resize_frame(frame)
            self.last_processed_frame = processed_frame
            self.last_scale = scale
            return processed_frame, True, scale
        else:
            # reutiliza frame anterior
            if self.last_processed_frame is not None:
                return self.last_processed_frame, False, self.last_scale
            else:
                # primeiro frame sempre processa
                processed_frame, scale = self.resize_frame(frame)
                self.last_processed_frame = processed_frame
                self.last_scale = scale
                return processed_frame, True, scale
    
    def get_stats(self) -> dict:
        """retorna estatisticas de processamento"""
        total = self.frames_processed + self.frames_skipped
        skip_ratio = (self.frames_skipped / total * 100) if total > 0 else 0
        
        return {
            'total_frames': total,
            'processed': self.frames_processed,
            'skipped': self.frames_skipped,
            'skip_ratio': skip_ratio,
            'resize_enabled': self.resize_enabled,
            'target_resolution': f"{self.target_width}x{self.target_height}"
        }
    
    def log_stats(self):
        """loga estatisticas"""
        stats = self.get_stats()
        logger.info("=== preprocessor stats ===")
        logger.info(f"total frames: {stats['total_frames']}")
        logger.info(f"processed: {stats['processed']}")
        logger.info(f"skipped: {stats['skipped']} ({stats['skip_ratio']:.1f}%)")
        if stats['resize_enabled']:
            logger.info(f"target resolution: {stats['target_resolution']}")


class AdaptivePreprocessor(FramePreprocessor):
    """preprocessador adaptativo baseado em carga"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processing_times = []
        self.max_processing_time = 100  # ms
        self.adaptive_skip = False
    
    def update_processing_time(self, processing_time_ms: float):
        """atualiza tempo de processamento"""
        self.processing_times.append(processing_time_ms)
        
        # mantem apenas ultimos 30 frames
        if len(self.processing_times) > 30:
            self.processing_times.pop(0)
        
        # ajusta skip automaticamente
        avg_time = np.mean(self.processing_times)
        if avg_time > self.max_processing_time:
            # muito lento, aumenta skip
            if self.frame_skip < 4:
                self.frame_skip += 1
                logger.warning(f"processamento lento ({avg_time:.1f}ms), aumentando skip para 1/{self.frame_skip}")
        elif avg_time < self.max_processing_time * 0.6:
            # rapido suficiente, diminui skip
            if self.frame_skip > 1:
                self.frame_skip -= 1
                logger.info(f"processamento rapido ({avg_time:.1f}ms), reduzindo skip para 1/{self.frame_skip}")
