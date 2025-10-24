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
    
    print(f"[DRAW_TEXTS] Drawing {len(values)} texts")
    for i, (text, x, y, color, use_small_font) in enumerate(values):
        print(f"  [{i}] '{text}' at ({x}, {y})")
    
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
        weapon_track_active = False  # indica se temos weapon track ativo
        
        # obtem confianca correta baseado no modo de votacao
        weapon_conf = 0.0
        if has_weapon and hasattr(track, 'weapon_classifier'):
            if hasattr(track.weapon_classifier, 'use_temporal_voting') and not track.weapon_classifier.use_temporal_voting:
                # modo sem votacao temporal - usa confianca atual
                weapon_conf = track.weapon_classifier.current_confidence
            else:
                # modo com votacao temporal - usa confianca acumulada
                weapon_conf = track.weapon_classifier.categories['armed']['confidence']
        
        # usa weapon_tracks com kalman filter se disponivel, senao usa weapon_bboxes raw
        weapon_bboxes = []
        max_weapon_conf = 0.0  # track da maior confianca dos weapon tracks
        
        if hasattr(track, 'weapon_tracks') and track.weapon_tracks:
            # usa bboxes suavizados pelo kalman filter
            for wt in track.weapon_tracks:
                # inclui weapon bboxes apenas se track nao estiver perdido
                if not wt.lost:
                    weapon_bboxes.append({
                        'bbox': wt.get_bbox('xyxy'),
                        'confidence': wt.confidence,
                        'class': wt.weapon_class,
                        'smoothed': True
                    })
                
                # mas considera a confianca MESMO se track estiver perdido
                # (para manter exibicao do texto de confianca)
                if wt.confidence > max_weapon_conf:
                    max_weapon_conf = wt.confidence
            
            # se temos weapon tracks (mesmo perdidos) com confianca, mostra arma
            if max_weapon_conf > 0:
                has_weapon = True  # força exibição
                weapon_conf = max_weapon_conf
                weapon_track_active = True  # track ativo
                
        elif hasattr(track, 'weapon_bboxes') and track.weapon_bboxes:
            # fallback para bboxes raw
            for wb in track.weapon_bboxes:
                if isinstance(wb, dict):
                    weapon_bboxes.append({
                        'bbox': wb.get('bbox', wb.get('bbox_crop', [])),
                        'confidence': wb.get('confidence', 0.0),
                        'class': wb.get('class', 'weapon'),
                        'smoothed': False
                    })
                else:
                    weapon_bboxes.append({
                        'bbox': wb,
                        'confidence': 0.0,
                        'class': 'weapon',
                        'smoothed': False
                    })
        
        # NOVO: Se a pessoa foi classificada como armada antes, mas agora o track foi perdido,
        # ainda mostra como armada com a ultima confianca conhecida (mas marca como "perdido")
        weapon_lost = False
        
        # Logica simplificada: se mostramos a arma (has_weapon=True) mas nao ha deteccao ativa (active=0)
        # entao o weapon track foi perdido e estamos apenas propagando a classificacao anterior
        if has_weapon and hasattr(track, 'weapon_tracks'):
            active_tracks = sum(1 for wt in track.weapon_tracks if wt.frames_since_update == 0)
            total_tracks = len(track.weapon_tracks)
            frames_info = [wt.frames_since_update for wt in track.weapon_tracks]
            
            # Se tem weapon tracks mas nenhum esta recebendo deteccao ativa neste frame
            weapon_lost = (total_tracks > 0 and active_tracks == 0)
            
            print(f"[DEBUG] Track {track.id}: has_weapon={has_weapon}, tracks={total_tracks} (active={active_tracks}, frames={frames_info}), weapon_lost={weapon_lost}, conf={weapon_conf:.2f}")
        
        return {
            'track_id': track.id,
            'bbox': bbox,
            'has_weapon': has_weapon,
            'weapon_confidence': weapon_conf,
            'weapon_lost': weapon_lost,  # indica se weapon track foi perdido
            'distance': track.distance if hasattr(track, 'distance') else None,
            'bearing': track.bearing if hasattr(track, 'bearing') else None,
            'lat': track.lat if hasattr(track, 'lat') else None,
            'lon': track.lon if hasattr(track, 'lon') else None,
            'x_utm': track.x_utm if hasattr(track, 'x_utm') else None,
            'y_utm': track.y_utm if hasattr(track, 'y_utm') else None,
            'weapon_bboxes': weapon_bboxes,
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
            'weapon_lost': track.get('weapon_lost', False),
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
    #bearing = track_data['bearing']
    lat = track_data['lat']
    lon = track_data['lon']
    #x_utm = track_data['x_utm']
    #y_utm = track_data['y_utm']
    has_weapon = track_data['has_weapon']
    weapon_conf = track_data['weapon_confidence']
    weapon_lost = track_data.get('weapon_lost', False)
    
    # Debug
    if has_weapon:
        print(f"[DRAW DEBUG] Track {track_id}: has_weapon={has_weapon}, weapon_lost={weapon_lost}, conf={weapon_conf:.2f}")
    
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
    #if bearing is not None: num_lines += 1
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
    '''
    if bearing is not None:
        text_bearing = f"Bearing: {bearing:.1f}°"
        text_values.append([text_bearing, xmin + 10, line_y, color_text_body, False])
        line_y += font_size + 3
    '''
    # coordenadas geograficas (branco)
    if lat is not None and lon is not None:
        text_geo = f"Lat:{lat:.6f} Lon:{lon:.6f}"
        text_values.append([text_geo, xmin + 10, line_y, color_text_body, False])
        line_y += font_size + 3
    
    # confianca da arma (vermelho claro se detectada)
    if has_weapon:
        if weapon_lost:
            # Track perdido - mostra com indicador de incerteza
            text_weapon = f"Confidence: {int(weapon_conf*100)}% (?)"
        else:
            # Track ativo - confianca normal
            text_weapon = f"Confidence: {int(weapon_conf*100)}%"
        text_values.append([text_weapon, xmin + 10, line_y, color_text_weapon, False])
        print(f"[TEXT ADDED] {track_id}: '{text_weapon}' at y={line_y}, has_weapon={has_weapon}, weapon_lost={weapon_lost}")
        line_y += font_size + 3
    
    return image, text_values


def _draw_weapon_bbox(image, weapon_bboxes):
    """desenha bbox vermelha para armas (sem texto)"""
    
    drawn_bboxes = []  # evita duplicatas
    
    for weapon_info in weapon_bboxes:
        try:
            if isinstance(weapon_info, dict):
                weapon_bbox = weapon_info.get('bbox', [])
            else:
                weapon_bbox = weapon_info
            
            wx1, wy1, wx2, wy2 = map(int, weapon_bbox)
            
            # verifica se ja desenhamos essa bbox (evita duplicatas)
            bbox_tuple = (wx1, wy1, wx2, wy2)
            if bbox_tuple in drawn_bboxes:
                continue
            drawn_bboxes.append(bbox_tuple)
            
        except (ValueError, TypeError, IndexError):
            continue
        
        # desenha bbox vermelha para arma (sempre vermelha, sem texto)
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
        print(f"[BBOX DEBUG] Track {track_data['track_id']}: {len(person_texts)} texts added, total now: {len(text_values)}")
        
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
