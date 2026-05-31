import cv2
import time
from PyQt6.QtCore import QThread, pyqtSignal
from camera_manager import open_camera_source
from constants import PROCESS_INTERVAL, is_any_model_enabled

# Lazy import de VideoProcessor (pode falhar se onnxruntime não instalado)
VideoProcessor = None


class VideoWorker(QThread):
    frame_signal = pyqtSignal(object)  # Emite frame processado
    fps_signal = pyqtSignal(float)
    info_signal = pyqtSignal(dict)  # {width, height, total_frames}
    processor_signal = pyqtSignal(dict)  # Emite metadados do processamento

    def __init__(self, source, enable_processing=True):
        super().__init__()
        self.source = source
        self.running = True
        self.paused = False
        self.fps = 0
        self._fps_count = 0
        self._fps_time = time.time()
        
        # Processamento ONNX
        self.enable_processing = enable_processing
        self.processor = None
        self.frame_count = 0

    def run(self):
        cap = open_camera_source(self.source)

        if not cap.isOpened():
            return

        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        if isinstance(self.source, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
            cap.set(cv2.CAP_PROP_FPS, 30)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.info_signal.emit({
            'width': width,
            'height': height,
            'total_frames': total_frames
        })
        
        # Inicializar processador se habilitado
        if self.enable_processing and is_any_model_enabled():
            try:
                global VideoProcessor
                if VideoProcessor is None:
                    from video_processor import VideoProcessor as VP
                    VideoProcessor = VP
                
                self.processor = VideoProcessor()
                print("✓ VideoProcessor inicializado com sucesso")
            except ImportError as e:
                if "onnxruntime" in str(e):
                    print("✗ onnxruntime não instalado, processamento desabilitado")
                else:
                    print(f"✗ Erro ao importar VideoProcessor: {e}")
                self.processor = None
            except Exception as e:
                print(f"✗ Erro ao inicializar VideoProcessor: {e}")
                self.processor = None

        while self.running and cap.isOpened():
            if not self.paused:
                ret, frame = cap.read()
                if not ret:
                    if isinstance(self.source, int):
                        time.sleep(0.05)
                        continue
                    break

                # Processar frame se habilitado e modelo disponível
                processed_frame = frame.copy()
                processing_metadata = {
                    "processed": False,
                    "frame_count": self.frame_count,
                }
                
                if self.processor is not None:
                    try:
                        if not self.processor.calibrated or self.frame_count % PROCESS_INTERVAL == 0:
                            result = self.processor.process_frame(frame)
                            processed_frame = result.get("frame", frame)
                            processing_metadata.update(result.get("metadata", {}))
                            processing_metadata.update(result.get("detections", {}))
                        else:
                            processed_frame = self.processor.transform_frame(frame)
                            processing_metadata.update({
                                "processed": True,
                                "calibrated": True,
                                "rotation_mode": self.processor.rotation_mode.name,
                                "ocr_text": self.processor.ocr_text_cache,
                                "pointer_upper": self.processor.upper_pointer_cache,
                                "pointer_lower": self.processor.lower_pointer_cache,
                            })

                        self.processor_signal.emit(processing_metadata)
                    except Exception as e:
                        print(f"✗ Erro ao processar frame: {e}")
                        processing_metadata["error"] = str(e)
                        self.processor_signal.emit(processing_metadata)
                
                self.frame_signal.emit(processed_frame)
                self.frame_count += 1
                
                self._fps_count += 1
                current_time = time.time()
                elapsed = current_time - self._fps_time
                
                if elapsed >= 1.0:
                    self.fps = self._fps_count / elapsed
                    self.fps_signal.emit(self.fps)
                    self._fps_count = 0
                    self._fps_time = current_time
            else:
                time.sleep(0.05)

        cap.release()

    def pause(self):
        """Pausar captura de frames"""
        self.paused = True

    def resume(self):
        """Retomar captura de frames"""
        self.paused = False

    def stop(self):
        """Parar thread completamente"""
        self.running = False
        self.quit()
        self.wait()
    
    def set_processing_enabled(self, enabled: bool):
        """Habilita ou desabilita processamento ONNX"""
        self.enable_processing = enabled
        if enabled and self.processor is None:
            try:
                self.processor = VideoProcessor()
                print("✓ VideoProcessor inicializado")
            except Exception as e:
                print(f"✗ Erro ao inicializar VideoProcessor: {e}")
        elif not enabled:
            self.processor = None
            print("Processamento ONNX desabilitado")
