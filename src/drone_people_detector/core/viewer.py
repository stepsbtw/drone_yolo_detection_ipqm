"""
visualizador profissional com texto de alta qualidade usando PIL
baseado no ship-detector-classifier
"""

import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import os
from pathlib import Path

# cores
color_text_body = (255, 255, 255)
color_text_title = (226, 135, 67)  # laranja
color_text_weapon = (255, 100, 100)  # vermelho claro

color_rect_person = (0, 180, 0)  # verde para pessoa
color_rect_weapon = (0, 0, 255)  # vermelho para arma
color_rect_bg = (0, 0, 0)  # fundo preto

color_pin = (67, 135, 226)  # azul
color_pin_weapon = (255, 100, 100)  # vermelho claro

min_rect_width = 400
show_bb_track = True

# fonte
font_size = 20
font = None
font_small = None

def _load_fonts():
    """carrega fontes ttf para texto de alta qualidade"""
    global font, font_small
    
    if font is not None:
        return
    
    # tenta encontrar arial.ttf
    font_paths = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
        '/System/Library/Fonts/Supplemental/Arial.ttf',
        'C:\\Windows\\Fonts\\arial.ttf',
    ]
    
    font_path = None
    for path in font_paths:
        if os.path.isfile(path):
            font_path = path
            break
    
    if font_path:
        try:
            font = ImageFont.truetype(font_path, size=font_size)
            font_small = ImageFont.truetype(font_path, size=16)
            return
        except Exception:
            pass
    
    # fallback para fonte default
    font = ImageFont.load_default()
    font_small = ImageFont.load_default()


def draw_texts(source_image, values):
    """desenha textos usando PIL para melhor qualidade"""
    _load_fonts()
    
    image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)
    
    for text, x, y, color, use_small_font in values:
        draw = ImageDraw.Draw(pil_image)
        current_font = font_small if use_small_font else font
        draw.text((x, y), text, fill=color, font=current_font)
    
    image = np.asarray(pil_image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def _extract_track_data(track):
    """extrai dados do track (suporta PersonTrack e dict)"""
    
    # objeto PersonTrack
    if hasattr(track, 'bbox') and not hasattr(track, 'get'):
        bbox_xywh = track.bbox
        if bbox_xywh is None or track.lost:
            return None
        
        x, y, w, h = bbox_xywh
        bbox = [x, y, x + w, y + h]
        
        has_weapon = track.weapon_classifier.has_weapon() if hasattr(track, 'weapon_classifier') else False
        weapon_conf = 0.0
        if has_weapon and hasattr(track, 'weapon_classifier'):
            weapon_conf = track.weapon_classifier.categories['armed']['confidence']
        
        return {
            'track_id': track.id,
            'bbox': bbox,
            'has_weapon': has_weapon,
            'weapon_confidence': weapon_conf,
            'distance': track.distance if hasattr(track, 'distance') else None,
            'bearing': track.bearing if hasattr(track, 'bearing') else None,
            'lat': track.lat if hasattr(track, 'lat') else None,
            'lon': track.lon if hasattr(track, 'lon') else None,
            'x_utm': track.x_utm if hasattr(track, 'x_utm') else None,
            'y_utm': track.y_utm if hasattr(track, 'y_utm') else None,
            'weapon_bboxes': track.weapon_bboxes if hasattr(track, 'weapon_bboxes') else [],
            'lost': track.lost
        }
    
    # formato dict
    elif isinstance(track, dict):
        bbox = track.get('bbox')
        if bbox is None:
            return None
        
        return {
            'track_id': track.get('track_id', 'Unknown'),
            'bbox': bbox,
            'has_weapon': track.get('has_weapon', False),
            'weapon_confidence': track.get('weapon_confidence', 0.0),
            'distance': track.get('distance'),
            'bearing': track.get('bearing'),
            'lat': track.get('lat'),
            'lon': track.get('lon'),
            'x_utm': track.get('x_utm'),
            'y_utm': track.get('y_utm'),
            'weapon_bboxes': track.get('weapon_bboxes', []),
            'lost': track.get('lost', False)
        }
    
    return None


def _draw_person_overlay(image, track_data):
    bbox = track_data['bbox']
    track_id = track_data['track_id']
    distance = track_data['distance']
    bearing = track_data['bearing']
    lat = track_data['lat']
    lon = track_data['lon']
    x_utm = track_data['x_utm']
    y_utm = track_data['y_utm']
    has_weapon = track_data['has_weapon']
    weapon_conf = track_data['weapon_confidence']
    
    try:
        x1, y1, x2, y2 = map(int, bbox)
        w = x2 - x1
        h = y2 - y1
    except (ValueError, TypeError, IndexError):
        return image, []
    
    text_values = []
    
    # desenha bbox da pessoa (opcional)
    if show_bb_track:
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color_rect_person, 2)
    
    # calcula posicao do info box
    xmin = int(x1 + w/4)
    
    # titulo
    text_title = f"ID: {track_id}"
    
    # conta quantas linhas precisamos
    num_lines = 1  # titulo
    if distance is not None: num_lines += 1
    if bearing is not None: num_lines += 1
    if lat is not None and lon is not None: num_lines += 1
    if has_weapon: num_lines += 1
    
    # calcula largura do retangulo (aumentada para caber mais info)
    rect_width = max((font_size * 20 * 0.6) + 20, min_rect_width)
    xmax = int(xmin + rect_width)
    ymax = int(y1)
    
    # centro do retangulo e bbox
    xcenter_src = int(xmin + rect_width/2)
    xcenter_dst = int(x1 + w/2)
    ycenter = int(y1 + h/2)
    
    # altura do info box
    ymin = int(y1 - font_size * (num_lines + 1))
    
    # desenha retangulo de fundo preto
    image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color_rect_bg, -1)
    
    # desenha linha conectando ao bbox (pin)
    image = cv2.line(image, (xcenter_src, ymax), (xcenter_dst, ycenter), color_rect_person, 2)
    
    # desenha circulo no centro do bbox
    image = cv2.circle(image, (xcenter_dst, ycenter), 8, color_pin, -1)
    
    # textos
    line_y = ymin + 5
    
    # titulo (laranja)
    text_values.append([text_title, xmin + 10, line_y, color_text_title, False])
    line_y += font_size + 3
    
    # distancia (branco)
    if distance is not None:
        text_dist = f"Dist: {distance:.1f}m"
        text_values.append([text_dist, xmin + 10, line_y, color_text_body, False])
        line_y += font_size + 3
    
    # bearing (branco)
    if bearing is not None:
        text_bearing = f"Bearing: {bearing:.1f}°"
        text_values.append([text_bearing, xmin + 10, line_y, color_text_body, False])
        line_y += font_size + 3
    
    # coordenadas geograficas (branco)
    if lat is not None and lon is not None:
        text_geo = f"Lat:{lat:.6f} Lon:{lon:.6f}"
        text_values.append([text_geo, xmin + 10, line_y, color_text_body, False])
        line_y += font_size + 3
    
    # arma (vermelho claro se detectada)
    if has_weapon:
        text_weapon = f"ARMA {int(weapon_conf*100)}%"
        text_values.append([text_weapon, xmin + 10, line_y, color_text_weapon, False])
        line_y += font_size + 3
    
    return image, text_values


def _draw_weapon_bbox(image, weapon_bboxes):
    """desenha bbox separada para armas"""
    
    for weapon_bbox in weapon_bboxes:
        try:
            wx1, wy1, wx2, wy2 = map(int, weapon_bbox)
        except (ValueError, TypeError, IndexError):
            continue
        
        # bbox vermelha fina
        image = cv2.rectangle(image, (wx1, wy1), (wx2, wy2), color_rect_weapon, 2)
    
    return image


def _draw_bbox(frame, tracks):
    if frame is None or not tracks:
        return frame
    
    image = frame.copy()
    text_values = []
    
    # processa cada track
    for track in tracks:
        track_data = _extract_track_data(track)
        
        if track_data is None or track_data['lost']:
            continue
        
        # desenha overlay da pessoa
        image, person_texts = _draw_person_overlay(image, track_data)
        text_values.extend(person_texts)
        
        # desenha bbox de armas se disponivel
        if track_data['weapon_bboxes']:
            image = _draw_weapon_bbox(
                image,
                track_data['weapon_bboxes']
            )
    
    # desenha todos os textos com PIL (alta qualidade)
    if text_values:
        image = draw_texts(image, text_values)
    
    return image


def add_frame_info(image, frame_count, people_count, weapons_count, fps, camera_id="drone-01"):
    """adiciona informacoes do frame no canto superior esquerdo"""
    _load_fonts()
    
    info_lines = [
        f"Frame: {frame_count}",
        f"People: {people_count}",
        f"Weapons: {weapons_count}",
        f"FPS: {fps:.1f}",
        f"Camera: {camera_id}",
    ]
    
    # fundo preto semi-transparente
    overlay = image.copy()
    cv2.rectangle(overlay, (5, 5), (280, 160), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    
    # textos
    text_values = []
    for i, line in enumerate(info_lines):
        y = 25 + i * 28
        text_values.append([line, 15, y, (0, 255, 0), False])
    
    image = draw_texts(image, text_values)
    
    return image
