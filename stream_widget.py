from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QCheckBox
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, pyqtSignal
import cv2
import os
from datetime import datetime
import threading
import queue


class StreamWidget(QWidget):
    close_signal = pyqtSignal()
    
    def __init__(self, source, source_name="Stream"):
        super().__init__()
        
        self.source = source
        self.source_name = source_name
        self.is_paused = False
        self.is_recording = False
        self.video_writer = None
        self.recording_path = None
        self.record_queue = None
        self.record_thread = None
        self.processing_enabled = True
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)
        
        top_bar = QHBoxLayout()
        self.label_title = QLabel(f"📹 {source_name}")
        self.label_title.setStyleSheet("font-weight: bold; font-size: 18px;")
        self.label_status = QLabel("● Ativo")
        self.label_status.setStyleSheet("color: #00ff00; font-weight: bold; font-size: 15px;")
        self.label_fps = QLabel("FPS: --")
        self.label_fps.setStyleSheet("font-size: 15px;")
        
        top_bar.addWidget(self.label_title)
        top_bar.addStretch()
        top_bar.addWidget(self.label_fps)
        top_bar.addWidget(self.label_status)
        layout.addLayout(top_bar)
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: #000; border: 1px solid #444;")
        self.video_label.setMinimumSize(360, 260)
        self.video_label.setScaledContents(False)
        layout.addWidget(self.video_label)
        
        # Área de metadados de processamento
        metadata_bar = QHBoxLayout()
        self.label_ocr = QLabel("OCR: N/A")
        self.label_ocr.setStyleSheet("font-size: 12px; color: #888;")
        self.label_pointers = QLabel("Ponteiros: N/A")
        self.label_pointers.setStyleSheet("font-size: 12px; color: #888;")
        metadata_bar.addWidget(self.label_ocr)
        metadata_bar.addWidget(self.label_pointers)
        metadata_bar.addStretch()
        layout.addLayout(metadata_bar)
        
        control_bar = QHBoxLayout()
        self.btn_pause = QPushButton("⏸ Pausar")
        self.btn_pause.setMinimumHeight(48)
        self.btn_pause.setMinimumWidth(120)
        self.btn_pause.clicked.connect(self.toggle_pause)
        
        self.btn_record = QPushButton("⏺ Gravar")
        self.btn_record.setMinimumHeight(48)
        self.btn_record.setMinimumWidth(120)
        self.btn_record.clicked.connect(self.toggle_record)
        
        self.checkbox_processing = QCheckBox("🤖 Processar ONNX")
        self.checkbox_processing.setChecked(True)
        self.checkbox_processing.setMinimumHeight(48)
        self.checkbox_processing.stateChanged.connect(self.on_processing_toggle)
        
        self.btn_close = QPushButton("✕ Fechar")
        self.btn_close.setMinimumHeight(48)
        self.btn_close.setMinimumWidth(120)
        self.btn_close.clicked.connect(self.close_stream)
        
        control_bar.addWidget(self.btn_pause)
        control_bar.addWidget(self.btn_record)
        control_bar.addWidget(self.checkbox_processing)
        control_bar.addStretch()
        control_bar.addWidget(self.btn_close)
        layout.addLayout(control_bar)
        
        self.setLayout(layout)
        self.video_worker = None

    def set_worker(self, worker):
        self.video_worker = worker
        worker.frame_signal.connect(self.update_frame)
        worker.fps_signal.connect(self.update_fps)
        worker.info_signal.connect(self.on_video_info)
        # Conectar ao sinal de metadados de processamento se disponível
        if hasattr(worker, 'processor_signal'):
            worker.processor_signal.connect(self.on_processor_metadata)

    def update_frame(self, frame):
        # Enfileira frame para gravação em background se estiver gravando
        if self.is_recording and self.record_queue is not None:
            try:
                self.record_queue.put_nowait(frame.copy())
            except queue.Full:
                # drop frame if queue is full to avoid blocking UI
                pass
        
        h, w, ch = frame.shape
        bytes_per_line = ch * w

        image = QImage(
            frame.data,
            w,
            h,
            bytes_per_line,
            QImage.Format.Format_BGR888
        )

        pixmap = QPixmap.fromImage(image)
        if self.video_label.width() > 0 and self.video_label.height() > 0:
            pixmap = pixmap.scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

        self.video_label.setPixmap(pixmap)

    def _recording_worker(self, path, fourcc, fps, frame_size, q):
        writer = cv2.VideoWriter(path, fourcc, fps, frame_size)
        try:
            while True:
                frame = q.get()
                if frame is None:
                    break
                try:
                    writer.write(frame)
                except Exception:
                    pass
        finally:
            try:
                writer.release()
            except Exception:
                pass
        # sinaliza que thread terminou
        self.record_thread = None

    def update_fps(self, fps):
        self.label_fps.setText(f"FPS: {fps:.1f}")

    def on_video_info(self, info):
        self.video_info = info

    def toggle_pause(self):
        if not self.video_worker:
            return
        
        if self.is_paused:
            self.video_worker.resume()
            self.is_paused = False
            self.btn_pause.setText("⏸ Pausar")
            self.label_status.setText("● Ativo")
            self.label_status.setStyleSheet("color: #00ff00; font-weight: bold; font-size: 13px;")
        else:
            self.video_worker.pause()
            self.is_paused = True
            self.btn_pause.setText("▶ Retomar")
            self.label_status.setText("⏸ Pausado")
            self.label_status.setStyleSheet("color: #ffaa00; font-weight: bold; font-size: 13px;")

    def toggle_record(self):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        if not self.video_worker:
            return
        
        os.makedirs("recordings", exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"recordings/stream_{self.source_name}_{timestamp}.avi"
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        fps = getattr(self.video_worker, 'fps', 30) or 30
        frame_size = (320, 240)
        
        if hasattr(self, 'video_info'):
            frame_size = (self.video_info['width'], self.video_info['height'])
        
        # Inicia fila e thread de gravação para não bloquear a UI
        self.record_queue = queue.Queue(maxsize=120)
        self.record_thread = threading.Thread(
            target=self._recording_worker,
            args=(filename, fourcc, fps, frame_size, self.record_queue),
            daemon=True,
        )
        self.record_thread.start()
        self.recording_path = filename
        self.is_recording = True
        self.btn_record.setText("⏹ Parar")
        self.btn_record.setStyleSheet("background-color: #ff4444; color: white;")
        self.label_status.setText("🔴 Gravando")
        self.label_status.setStyleSheet("color: #ff0000; font-weight: bold; font-size: 13px;")

    def stop_recording(self):
        # Envia sentinel para thread de gravação e aguarda término
        if self.record_queue is not None:
            try:
                self.record_queue.put_nowait(None)
            except Exception:
                try:
                    self.record_queue.put(None, timeout=0.5)
                except Exception:
                    pass

        if self.record_thread is not None:
            self.record_thread.join(timeout=2.0)
            self.record_thread = None

        self.record_queue = None
        
        self.is_recording = False
        self.btn_record.setText("⏺ Gravar")
        self.btn_record.setStyleSheet("")
        self.label_status.setText("● Ativo")
        self.label_status.setStyleSheet("color: #00ff00; font-weight: bold; font-size: 13px;")

    def close_stream(self):
        if self.video_worker:
            self.video_worker.stop()
        
        if self.is_recording:
            self.stop_recording()
        
        self.close_signal.emit()
    
    def on_processor_metadata(self, metadata):
        """Atualiza labels com metadados de processamento ONNX."""
        try:
            if metadata.get("processed"):
                ocr_text = metadata.get("ocr_text", "N/A")
                self.label_ocr.setText(f"OCR: {ocr_text}")
                self.label_ocr.setStyleSheet("font-size: 12px; color: #0f0;")
                
                pointer_upper = metadata.get("pointer_upper", 0.0)
                pointer_lower = metadata.get("pointer_lower", 0.0)
                self.label_pointers.setText(f"Ponteiros: {pointer_upper} | {pointer_lower}")
                self.label_pointers.setStyleSheet("font-size: 12px; color: #0f0;")
            else:
                # Reset se não foi processado
                pass
        except Exception as e:
            print(f"Erro ao atualizar metadados: {e}")
    
    def on_processing_toggle(self, state):
        """Habilita/desabilita processamento ONNX."""
        if not self.video_worker:
            return
        
        enabled = self.checkbox_processing.isChecked()
        self.processing_enabled = enabled
        
        if hasattr(self.video_worker, 'set_processing_enabled'):
            self.video_worker.set_processing_enabled(enabled)
            status = "ativado" if enabled else "desativado"
            print(f"Processamento ONNX {status} para {self.source_name}")

    def cleanup(self):
        self.close_stream()
