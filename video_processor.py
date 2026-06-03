"""
VideoProcessor: Módulo centralizado para processamento de vídeo com modelos ONNX.

Responsável por:
- Carregar modelos ONNX conforme flags de habilitação
- Detectar rotação automática (primeira frame)
- Calibrar homografia (primeira frame)
- Processar frames com inferência ONNX
- Gerenciar estado (rotation, homography, calibração)
- Tratamento gracioso de erros (fallback se modelos falham)
"""

import cv2
import numpy as np
import math
from typing import Dict, Tuple, List, Optional
from enum import Enum

from constants import (
    MODELS_PATHS,
    CONFIG_MODELS_ENABLED,
    SEG_SIZE,
    YOLO_SIZE,
    THRESHOLD_UNET,
    CONF_THRESHOLD_YOLO,
    CONF_THRESHOLD_POSE,
    OCR_CHARS,
    ONNX_PROVIDERS,
    ONNX_PROVIDERS_CPU_ONLY,
)

# Lazy import de onnxruntime (será importado apenas se necessário)
ort = None


class RotationMode(Enum):
    """Enumeração de modos de rotação detectados automaticamente."""
    NONE = None
    ROTATE_90_CW = cv2.ROTATE_90_CLOCKWISE
    ROTATE_90_CCW = cv2.ROTATE_90_COUNTERCLOCKWISE
    ROTATE_180 = cv2.ROTATE_180


class VideoProcessor:
    """
    Processador de vídeo que gerencia modelos ONNX, rotação, homografia e inferência.
    
    Estados gerenciados:
    - calibrado: Se homografia foi calculada
    - rotation_mode: Tipo de rotação detectada
    - homography_matrix: Matriz de perspectiva calculada
    - target_dimensions: Dimensões finais do vídeo corrigido
    """

    def __init__(self, enabled_models: Optional[Dict[str, bool]] = None):
        """
        Inicializa o processador.
        
        Args:
            enabled_models: Dicionário com flags de modelos. Se None, usa CONFIG_MODELS_ENABLED
        """
        self.enabled_models = enabled_models or CONFIG_MODELS_ENABLED.copy()
        
        # Estado de calibração
        self.calibrated = False
        self.rotation_mode = RotationMode.NONE
        self.homography_matrix = None
        self.target_dimensions = None
        
        # Modelos ONNX (carregados conforme necessário)
        self.seg_visor = None
        self.yolo_detector = None
        self.ocr_session = None
        self.pose_session = None
        
        # Nomes das entradas dos modelos (cached)
        self.input_names = {}
        
        # Cache de resultados (para renderização contínua)
        self.ocr_text_cache = "N/A"
        self.upper_pointer_cache = 0.0
        self.lower_pointer_cache = 0.0
        self.pointer_boxes_cache = []
        
        # Carregar modelos habilitados
        self._load_models()
    
    def _load_models(self):
        """Carrega apenas os modelos que estão habilitados."""
        global ort
        
        # Lazy import de onnxruntime
        if ort is None:
            try:
                import onnxruntime as ort_module
                ort = ort_module
            except ImportError:
                print("✗ ERRO: onnxruntime não está instalado!")
                print("  Instale com: pip install onnxruntime")
                return
        
        print("\n" + "="*60)
        print("Carregando modelos ONNX habilitados...")
        print("="*60)
        
        try:
            if self.enabled_models.get("visor", False):
                path = MODELS_PATHS.get("visor")
                if path and self._file_exists(path):
                    self.seg_visor = ort.InferenceSession(
                        path, providers=ONNX_PROVIDERS
                    )
                    self.input_names["visor"] = self.seg_visor.get_inputs()[0].name
                    print(f"✓ U-Net Visor carregado: {path}")
                else:
                    print(f"✗ U-Net Visor não encontrado, desabilitando...")
                    self.enabled_models["visor"] = False
        except Exception as e:
            print(f"✗ Erro ao carregar U-Net Visor: {e}")
            self.enabled_models["visor"] = False
        
        try:
            if self.enabled_models.get("ponteiros", False):
                path = MODELS_PATHS.get("ponteiros")
                if path and self._file_exists(path):
                    self.yolo_detector = ort.InferenceSession(
                        path, providers=ONNX_PROVIDERS
                    )
                    self.input_names["ponteiros"] = self.yolo_detector.get_inputs()[0].name
                    print(f"✓ YOLO Detector carregado: {path}")
                else:
                    print(f"✗ YOLO Detector não encontrado, desabilitando...")
                    self.enabled_models["ponteiros"] = False
        except Exception as e:
            print(f"✗ Erro ao carregar YOLO Detector: {e}")
            self.enabled_models["ponteiros"] = False
        
        try:
            if self.enabled_models.get("ocr", False):
                path = MODELS_PATHS.get("ocr")
                if path and self._file_exists(path):
                    self.ocr_session = ort.InferenceSession(
                        path, providers=ONNX_PROVIDERS_CPU_ONLY
                    )
                    self.input_names["ocr"] = self.ocr_session.get_inputs()[0].name
                    print(f"✓ OCR Session carregado: {path}")
                else:
                    print(f"✗ OCR não encontrado, desabilitando...")
                    self.enabled_models["ocr"] = False
        except Exception as e:
            print(f"✗ Erro ao carregar OCR: {e}")
            self.enabled_models["ocr"] = False
        
        try:
            if self.enabled_models.get("pose", False):
                path = MODELS_PATHS.get("pose")
                if path and self._file_exists(path):
                    self.pose_session = ort.InferenceSession(
                        path, providers=ONNX_PROVIDERS
                    )
                    self.input_names["pose"] = self.pose_session.get_inputs()[0].name
                    print(f"✓ YOLO Pose carregado: {path}")
                else:
                    print(f"✗ YOLO Pose não encontrado, desabilitando...")
                    self.enabled_models["pose"] = False
        except Exception as e:
            print(f"✗ Erro ao carregar YOLO Pose: {e}")
            self.enabled_models["pose"] = False
        
        print("="*60 + "\n")
    
    @staticmethod
    def _file_exists(path: str) -> bool:
        """Verifica se arquivo existe."""
        import os
        return os.path.exists(path)
    
    @staticmethod
    def _letterbox(im, new_shape=(YOLO_SIZE, YOLO_SIZE), color=(114, 114, 114)):
        """Redimensiona imagem com padding (letterbox)."""
        shape = im.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return im, r, dw, dh
    
    def calibrate(self, frame: np.ndarray) -> bool:
        """
        Calibra a rotação e homografia baseado no primeiro frame.
        
        Args:
            frame: Frame para calibração (BGR)
            
        Returns:
            True se calibração bem-sucedida, False caso contrário
        """
        if self.calibrated:
            return True
        
        if not self.enabled_models.get("visor", False) or self.seg_visor is None:
            print("⚠️ Calibração interrompida: U-Net Visor não habilitado")
            return False
        
        if not self.enabled_models.get("ponteiros", False) or self.yolo_detector is None:
            print("⚠️ Calibração interrompida: YOLO Detector não habilitado")
            return False
        
        try:
            print("[Calibração] Calculando rotação e homografia...")
            h_orig, w_orig = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 1. U-Net para máscara do visor
            seg_inp = cv2.resize(frame_rgb, (SEG_SIZE, SEG_SIZE)).astype(np.float32) / 255.0
            seg_inp = np.transpose(seg_inp, (2, 0, 1))[None, ...]
            pred_seg = self.seg_visor.run(None, {self.input_names["visor"]: seg_inp})[0]
            mask_prob = 1 / (1 + np.exp(-pred_seg))
            mask_v = (np.squeeze(mask_prob) > THRESHOLD_UNET).astype(np.uint8)
            mask_v = cv2.resize(mask_v, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            
            # 2. Encontrar contorno maior (visor)
            contours_v, _ = cv2.findContours(mask_v, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours_v) == 0:
                print("⚠️ Falha na calibração: Visor não detectado no Frame 1")
                return False
            
            maior_cnt = max(contours_v, key=cv2.contourArea)
            x_b, y_b, w_b, h_b = cv2.boundingRect(maior_cnt)
            centro_x_visor = x_b + (w_b / 2)
            centro_y_visor = y_b + (h_b / 2)
            
            # 3. YOLO para ponteiros
            inp_img, ratio, dw, dh = self._letterbox(frame_rgb, new_shape=(YOLO_SIZE, YOLO_SIZE))
            yolo_inp = np.transpose(inp_img.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]
            saida_bruta_yolo = self.yolo_detector.run(None, {self.input_names["ponteiros"]: yolo_inp})[0]
            
            boxes_p = []
            for pred in saida_bruta_yolo[0]:
                if pred[4] < CONF_THRESHOLD_YOLO:
                    continue
                boxes_p.append([
                    int(max(0, (pred[0] - dw) / ratio)),
                    int(max(0, (pred[1] - dh) / ratio)),
                    int(min(w_orig, (pred[2] - dw) / ratio)),
                    int(min(h_orig, (pred[3] - dh) / ratio))
                ])
            
            if len(boxes_p) == 0:
                print("⚠️ Falha na calibração: Ponteiros não detectados no Frame 1")
                return False
            
            centro_x_ponteiros = np.mean([(b[0] + b[2]) / 2 for b in boxes_p])
            centro_y_ponteiros = np.mean([(b[1] + b[3]) / 2 for b in boxes_p])
            
            # 4. Determinar rotação ortogonal
            frame_calib = frame.copy()
            mask_calib = mask_v.copy()
            
            if h_b > w_b:  # Visor em pé
                if centro_x_ponteiros > centro_x_visor:
                    frame_calib = cv2.rotate(frame_calib, cv2.ROTATE_90_CLOCKWISE)
                    mask_calib = cv2.rotate(mask_calib, cv2.ROTATE_90_CLOCKWISE)
                    self.rotation_mode = RotationMode.ROTATE_90_CW
                elif centro_x_ponteiros < centro_x_visor:
                    frame_calib = cv2.rotate(frame_calib, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    mask_calib = cv2.rotate(mask_calib, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    self.rotation_mode = RotationMode.ROTATE_90_CCW
            else:  # Visor deitado
                if centro_y_visor > centro_y_ponteiros:
                    frame_calib = cv2.rotate(frame_calib, cv2.ROTATE_180)
                    mask_calib = cv2.rotate(mask_calib, cv2.ROTATE_180)
                    self.rotation_mode = RotationMode.ROTATE_180
            
            h_c, w_c = frame_calib.shape[:2]
            
            # 5. Ajuste fino angular
            contours_c, _ = cv2.findContours(mask_calib, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours_c) == 0:
                print("⚠️ Falha no ajuste fino: Nenhum contorno após rotação")
                self.calibrated = True
                return True
            
            maior_cnt_c = max(contours_c, key=cv2.contourArea)
            rect = cv2.minAreaRect(maior_cnt_c)
            (cx_c, cy_c), (l_rect, a_rect), angulo = rect
            
            if l_rect < a_rect:
                angulo = angulo - 90 if angulo > 0 else angulo + 90
            else:
                if angulo > 45:
                    angulo -= 90
                elif angulo < -45:
                    angulo += 90
            
            if abs(angulo) > 0.1:
                M_rot = cv2.getRotationMatrix2D((cx_c, cy_c), angulo, 1.0)
                mask_calib = cv2.warpAffine(mask_calib, M_rot, (w_c, h_c), flags=cv2.INTER_NEAREST)
                contours_c, _ = cv2.findContours(mask_calib, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours_c) > 0:
                    maior_cnt_c = max(contours_c, key=cv2.contourArea)
            
            # 6. Construir homografia
            rect = cv2.minAreaRect(maior_cnt_c)
            box = cv2.boxPoints(rect).astype(np.intp)
            
            pts_origem = np.zeros((4, 2), dtype="float32")
            s = box.sum(axis=1)
            pts_origem[0] = box[np.argmin(s)]
            pts_origem[2] = box[np.argmax(s)]
            diff = np.diff(box, axis=1)
            pts_origem[1] = box[np.argmin(diff)]
            pts_origem[3] = box[np.argmax(diff)]
            
            dim1, dim2 = rect[1]
            largura_visor = int(max(dim1, dim2))
            altura_visor = int(min(dim1, dim2))
            
            margem_w = int(largura_visor * 1.2)
            margem_h = int(altura_visor * 4.5)
            
            pts_destino = np.array([
                [margem_w, margem_h],
                [margem_w + largura_visor, margem_h],
                [margem_w + largura_visor, margem_h + altura_visor],
                [margem_w, margem_h + altura_visor]
            ], dtype="float32")
            
            self.homography_matrix = cv2.getPerspectiveTransform(pts_origem, pts_destino)
            self.target_dimensions = (largura_visor + margem_w * 2, altura_visor + margem_h * 2)
            self.calibrated = True
            
            print(f"[Calibração] ✓ Pronto! Rotação: {self.rotation_mode.name}, Homografia calculada")
            return True
            
        except Exception as e:
            print(f"✗ Erro na calibração: {e}")
            return False
    
    def process_frame(self, frame: np.ndarray, force_calibration: bool = False) -> Dict:
        """
        Processa um frame aplicando transformações e inferência ONNX.
        
        Args:
            frame: Frame para processar (BGR)
            force_calibration: Se True, força recalibração
            
        Returns:
            Dicionário com:
            - "frame": frame processado
            - "detections": detecções (ponteiros, visor)
            - "metadata": metadados (ocr_text, pointer_values, rotation, etc)
        """
        result = {
            "frame": frame.copy(),
            "detections": {},
            "metadata": {
                "processed": False,
                "calibrated": self.calibrated,
                "rotation_mode": self.rotation_mode.name if self.rotation_mode else None,
            }
        }
        
        # Se nenhum modelo habilitado, retorna frame bruto
        if not any(self.enabled_models.values()):
            result["metadata"]["processed"] = False
            return result
        
        try:
            # Calibrar se necessário
            if not self.calibrated or force_calibration:
                if not self.calibrate(frame):
                    return result

            result["frame"] = self.transform_frame(frame)
            result["metadata"]["processed"] = True
            
            # Processar visor (OCR)
            if self.enabled_models.get("ocr", False) and self.ocr_session is not None:
                self._process_ocr(result["frame"], result)
            
            # Detectar ponteiros e calcular valores
            if self.enabled_models.get("ponteiros", False) and self.yolo_detector is not None:
                self._process_pointers(result["frame"], result)
            
            return result
            
        except Exception as e:
            print(f"✗ Erro ao processar frame: {e}")
            result["frame"] = frame
            result["metadata"]["processed"] = False
            return result
    
    def transform_frame(self, frame: np.ndarray) -> np.ndarray:
        """Aplica rotação e homografia a um frame usando o estado de calibração atual."""
        frame_rot = frame.copy()
        if self.rotation_mode != RotationMode.NONE:
            frame_rot = cv2.rotate(frame_rot, self.rotation_mode.value)

        if self.homography_matrix is not None and self.target_dimensions is not None:
            return cv2.warpPerspective(frame_rot, self.homography_matrix, self.target_dimensions)

        return frame_rot

    def _process_ocr(self, frame: np.ndarray, result: Dict):
        """Processa OCR no visor."""
        try:
            if not self.enabled_models.get("visor", False) or self.seg_visor is None:
                return
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            
            # U-Net para máscara
            seg_inp = cv2.resize(frame_rgb, (SEG_SIZE, SEG_SIZE)).astype(np.float32) / 255.0
            seg_inp = np.transpose(seg_inp, (2, 0, 1))[None, ...]
            pred_seg = self.seg_visor.run(None, {self.input_names["visor"]: seg_inp})[0]
            mask_prob = 1 / (1 + np.exp(-pred_seg))
            mask_v = (np.squeeze(mask_prob) > THRESHOLD_UNET).astype(np.uint8)
            mask_v = cv2.resize(mask_v, (w, h), interpolation=cv2.INTER_NEAREST)
            
            contours, _ = cv2.findContours(mask_v, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                return
            
            c = max(contours, key=cv2.contourArea)
            x_v, y_v, ww_v, hh_v = cv2.boundingRect(c)
            crop_visor = frame[y_v:y_v + hh_v, x_v:x_v + ww_v]
            
            if crop_visor.size == 0:
                return
            
            crop_gray = cv2.cvtColor(crop_visor, cv2.COLOR_BGR2GRAY)
            crop_resized = cv2.resize(crop_gray, (128, 32), interpolation=cv2.INTER_LINEAR)
            tensor_ocr = np.expand_dims(crop_resized.astype(np.float32) / 255.0, axis=(0, 1))
            
            pred_ocr = self.ocr_session.run(None, {self.input_names["ocr"]: tensor_ocr})[0]
            pred_argmax = np.argmax(pred_ocr, axis=2).flatten()
            
            texto_ocr = ""
            last = -1
            for p in pred_argmax:
                if p != last and p != 0 and (p - 1) < len(OCR_CHARS):
                    texto_ocr += OCR_CHARS[p - 1]
                last = p
            
            self.ocr_text_cache = texto_ocr
            result["metadata"]["ocr_text"] = texto_ocr
            result["detections"]["visor_bbox"] = (x_v, y_v, ww_v, hh_v)
            
        except Exception as e:
            print(f"✗ Erro no OCR: {e}")
    
    def _process_pointers(self, frame: np.ndarray, result: Dict):
        """Detecta ponteiros e calcula valores analógicos."""
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            
            # YOLO detecção
            inp_img, ratio, dw, dh = self._letterbox(frame_rgb, new_shape=(YOLO_SIZE, YOLO_SIZE))
            yolo_inp = np.transpose(inp_img.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]
            saida_bruta_yolo = self.yolo_detector.run(None, {self.input_names["ponteiros"]: yolo_inp})[0]
            
            ponteiros_validos = []
            for pred in saida_bruta_yolo[0]:
                if pred[4] < CONF_THRESHOLD_YOLO:
                    continue
                ponteiros_validos.append([
                    int(max(0, (pred[0] - dw) / ratio)),
                    int(max(0, (pred[1] - dh) / ratio)),
                    int(min(w, (pred[2] - dw) / ratio)),
                    int(min(h, (pred[3] - dh) / ratio))
                ])
            
            self.pointer_boxes_cache = sorted(ponteiros_validos, key=lambda p: p[1])
            valores_analogicos = []
            
            # Processar cada ponteiro com YOLO Pose
            if self.enabled_models.get("pose", False) and self.pose_session is not None:
                for idx, (x1, y1, x2, y2) in enumerate(self.pointer_boxes_cache):
                    crop_p = frame[y1:y2, x1:x2]
                    if crop_p.size == 0:
                        valores_analogicos.append(0.0)
                        continue
                    
                    crop_rgb = cv2.cvtColor(crop_p, cv2.COLOR_BGR2RGB)
                    inp_pose, ratio_p, dw_p, dh_p = self._letterbox(crop_rgb, new_shape=(YOLO_SIZE, YOLO_SIZE))
                    inp_p_tensor = inp_pose.astype(np.float32) / 255.0
                    inp_p_tensor = np.transpose(inp_p_tensor, (2, 0, 1))[None, ...]
                    
                    saida_pose = self.pose_session.run(None, {self.input_names["pose"]: inp_p_tensor})
                    preds_pose = np.transpose(saida_pose[0][0])
                    
                    melhor_score = 0.0
                    melhor_pred = None
                    
                    for pred in preds_pose:
                        score = pred[4]
                        if score > melhor_score and score >= CONF_THRESHOLD_POSE:
                            melhor_score = score
                            melhor_pred = pred
                    
                    if melhor_pred is not None:
                        kpts = melhor_pred[5:]
                        kpt0_x, kpt0_y, kpt0_conf = kpts[0], kpts[1], kpts[2]
                        kpt1_x, kpt1_y, kpt1_conf = kpts[3], kpts[4], kpts[5]
                        
                        if kpt0_conf > 0.5 and kpt1_conf > 0.5:
                            eixo_x = (kpt0_x - dw_p) / ratio_p
                            eixo_y = (kpt0_y - dh_p) / ratio_p
                            ponta_x = (kpt1_x - dw_p) / ratio_p
                            ponta_y = (kpt1_y - dh_p) / ratio_p
                            
                            dx = ponta_x - eixo_x
                            dy = eixo_y - ponta_y
                            
                            angulo_ajustado = (90 - math.degrees(math.atan2(dy, dx))) % 360
                            valores_analogicos.append(round(angulo_ajustado / 36.0, 1))
                        else:
                            valores_analogicos.append(0.0)
                    else:
                        valores_analogicos.append(0.0)
            
            self.upper_pointer_cache = valores_analogicos[0] if len(valores_analogicos) > 0 else 0.0
            self.lower_pointer_cache = valores_analogicos[1] if len(valores_analogicos) > 1 else 0.0
            
            result["detections"]["pointer_boxes"] = self.pointer_boxes_cache
            result["metadata"]["pointer_values"] = valores_analogicos
            result["metadata"]["pointer_upper"] = self.upper_pointer_cache
            result["metadata"]["pointer_lower"] = self.lower_pointer_cache
            
        except Exception as e:
            print(f"✗ Erro ao processar ponteiros: {e}")
    
    def get_status(self) -> Dict:
        """Retorna status atual do processador."""
        return {
            "calibrated": self.calibrated,
            "rotation_mode": self.rotation_mode.name if self.rotation_mode else None,
            "homography_ready": self.homography_matrix is not None,
            "enabled_models": self.enabled_models.copy(),
            "ocr_text": self.ocr_text_cache,
            "pointer_upper": self.upper_pointer_cache,
            "pointer_lower": self.lower_pointer_cache,
        }
    
    def reset_calibration(self):
        """Reseta calibração (força recalibração no próximo frame)."""
        self.calibrated = False
        self.rotation_mode = RotationMode.NONE
        self.homography_matrix = None
        self.target_dimensions = None
        print("Calibração resetada")
    
    def enable_model(self, model_name: str):
        """Habilita um modelo específico."""
        if model_name in self.enabled_models:
            self.enabled_models[model_name] = True
            print(f"Modelo '{model_name}' habilitado")
    
    def disable_model(self, model_name: str):
        """Desabilita um modelo específico."""
        if model_name in self.enabled_models:
            self.enabled_models[model_name] = False
            print(f"Modelo '{model_name}' desabilitado")
