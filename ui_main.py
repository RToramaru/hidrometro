from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QGridLayout,
    QPushButton, QFileDialog, QHBoxLayout, QMessageBox, QInputDialog,
    QDialog, QCheckBox, QSpinBox, QLabel, QGroupBox
)
from PyQt6.QtCore import QTimer
from stream_widget import StreamWidget
from camera_manager import list_cameras
from video_worker import VideoWorker
from constants import CONFIG_MODELS_ENABLED, PROCESS_INTERVAL
import os


# ============================================
# TEMAS CSS
# ============================================
DARK_THEME = """
    QMainWindow {{
        background-color: #1e1e1e;
        color: #ffffff;
    }}
    QWidget {{
        background-color: #1e1e1e;
        color: #ffffff;
    }}
    QPushButton {{
        background-color: #0078d4;
        color: white;
        border: none;
        padding: 12px 20px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 16px;
        min-height: 54px;
        min-width: 180px;
    }}
    QPushButton:hover {{
        background-color: #005a9e;
    }}
    QPushButton:pressed {{
        background-color: #004b50;
    }}
    QLabel {{
        color: #ffffff;
        font-size: 14px;
    }}
    QWidget#topBar {
        background-color: #2b2b2b;
        border: 1px solid #444;
        border-radius: 12px;
        padding: 8px;
    }
"""

LIGHT_THEME = """
    QMainWindow {{
        background-color: #f5f5f5;
        color: #000000;
    }}
    QWidget {{
        background-color: #f5f5f5;
        color: #000000;
    }}
    QPushButton {{
        background-color: #0078d4;
        color: white;
        border: none;
        padding: 12px 20px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 14px;
        min-height: 44px;
    }}
    QPushButton:hover {{
        background-color: #005a9e;
    }}
    QPushButton:pressed {{
        background-color: #004b50;
    }}
    QLabel {{
        color: #000000;
        font-size: 14px;
    }}
    QWidget#topBar {
        background-color: #e5e5e5;
        border: 1px solid #b0b0b0;
        border-radius: 12px;
        padding: 8px;
    }
"""


class OnnxConfigDialog(QDialog):
    """Dialog para configurar flags e parâmetros dos modelos ONNX."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("⚙️ Configuração de Modelos ONNX")
        self.setGeometry(100, 100, 500, 400)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        
        # Título
        title = QLabel("Configurar Modelos ONNX")
        title.setStyleSheet("font-weight: bold; font-size: 16px;")
        layout.addWidget(title)
        
        # Grupo de checkboxes para modelos
        models_group = QGroupBox("Habilitar/Desabilitar Modelos")
        models_layout = QVBoxLayout()
        
        self.checkboxes = {}
        for model_name in ["visor", "ponteiros", "ocr", "pose"]:
            checkbox = QCheckBox(f"🤖 {model_name.capitalize()}")
            checkbox.setChecked(CONFIG_MODELS_ENABLED.get(model_name, True))
            self.checkboxes[model_name] = checkbox
            models_layout.addWidget(checkbox)
        
        models_group.setLayout(models_layout)
        layout.addWidget(models_group)
        
        # Config de performance
        perf_group = QGroupBox("Configuração de Performance")
        perf_layout = QVBoxLayout()
        
        # Label + SpinBox para PROCESS_INTERVAL
        interval_layout = QHBoxLayout()
        interval_label = QLabel("Processar a cada N frames:")
        self.spinbox_interval = QSpinBox()
        self.spinbox_interval.setMinimum(1)
        self.spinbox_interval.setMaximum(30)
        self.spinbox_interval.setValue(PROCESS_INTERVAL)
        self.spinbox_interval.setSuffix(" frames")
        interval_layout.addWidget(interval_label)
        interval_layout.addStretch()
        interval_layout.addWidget(self.spinbox_interval)
        perf_layout.addLayout(interval_layout)
        
        # Informação
        info_label = QLabel(
            "Quanto maior o número, menos frequente o processamento.\n"
            "Recomendado: 3-5 frames para manter FPS bom."
        )
        info_label.setStyleSheet("font-size: 11px; color: #666;")
        perf_layout.addWidget(info_label)
        
        perf_group.setLayout(perf_layout)
        layout.addWidget(perf_group)
        
        layout.addStretch()
        
        # Botões
        buttons_layout = QHBoxLayout()
        
        btn_apply = QPushButton("✓ Aplicar")
        btn_apply.clicked.connect(self.apply_config)
        
        btn_cancel = QPushButton("✕ Cancelar")
        btn_cancel.clicked.connect(self.reject)
        
        buttons_layout.addStretch()
        buttons_layout.addWidget(btn_apply)
        buttons_layout.addWidget(btn_cancel)
        layout.addLayout(buttons_layout)
    
    def apply_config(self):
        """Aplicar configurações."""
        # Atualizar CONFIG_MODELS_ENABLED
        for model_name, checkbox in self.checkboxes.items():
            CONFIG_MODELS_ENABLED[model_name] = checkbox.isChecked()
        
        # Atualizar PROCESS_INTERVAL (nota: é constante, então atualizamos direto em constants)
        import constants
        constants.PROCESS_INTERVAL = self.spinbox_interval.value()
        
        print("✓ Configuração ONNX aplicada:")
        print(f"  Modelos habilitados: {[m for m, e in CONFIG_MODELS_ENABLED.items() if e]}")
        print(f"  Intervalo de processamento: {constants.PROCESS_INTERVAL} frames")
        
        self.accept()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("🎥 Multi Viewer - Câmeras & Vídeos")
        self.resize(1400, 900)

        self.central = QWidget()
        self.setCentralWidget(self.central)

        self.main_layout = QVBoxLayout(self.central)
        self.main_layout.setContentsMargins(12, 12, 12, 12)
        self.main_layout.setSpacing(12)

        self.grid_layout = QGridLayout()
        self.grid_layout.setSpacing(12)

        self.streams = []
        self.workers = []
        self.dark_mode = False
        
        self.layout_timer = QTimer()
        self.layout_timer.timeout.connect(self._do_relayout)
        self.layout_timer.setSingleShot(True)

        # ============================================
        # BARRA SUPERIOR
        # ============================================
        top_bar_widget = QWidget()
        top_bar_widget.setObjectName("topBar")
        top_bar = QHBoxLayout(top_bar_widget)
        top_bar.setContentsMargins(0, 0, 0, 0)
        top_bar.setSpacing(18)
        
        self.btn_camera = QPushButton("📷 Adicionar Câmera")
        self.btn_camera.clicked.connect(self.add_camera_menu)
        self.btn_camera.setMinimumSize(180, 52)
        
        self.btn_video = QPushButton("🎬 Adicionar Vídeo")
        self.btn_video.clicked.connect(self.add_video)
        self.btn_video.setMinimumSize(180, 52)
        
        self.btn_onnx_config = QPushButton("⚙️ Modelos ONNX")
        self.btn_onnx_config.clicked.connect(self.open_onnx_config)
        self.btn_onnx_config.setMinimumSize(180, 52)

        top_bar.addWidget(self.btn_camera)
        top_bar.addWidget(self.btn_video)
        top_bar.addWidget(self.btn_onnx_config)
        top_bar.addStretch()

        self.main_layout.addWidget(top_bar_widget)
        self.main_layout.addLayout(self.grid_layout)

        # salvar referência antes de aplicar tema
        self.top_bar_widget = top_bar_widget

        self.apply_theme(dark=False)

    def relayout_streams(self):
        if self.layout_timer.isActive():
            self.layout_timer.stop()
        self.layout_timer.start(10)

    def _clear_grid(self):
        while self.grid_layout.count() > 0:
            item = self.grid_layout.takeAt(0)
            if item.widget():
                self.grid_layout.removeWidget(item.widget())

    def _do_relayout(self):
        self._clear_grid()

        total = len(self.streams)
        if total == 0:
            return

        cols = int(total ** 0.5)
        if cols * cols < total:
            cols += 1

        screen = self.screen().availableGeometry()
        max_w = screen.width()
        max_h = screen.height() - 140

        tile_w = max_w // cols
        rows = (total + cols - 1) // cols
        tile_h = max_h // rows

        for idx, widget in enumerate(self.streams):
            widget.setMinimumSize(tile_w - 16, tile_h - 16)
            widget.setMaximumSize(tile_w, tile_h)
            row = idx // cols
            col = idx % cols
            self.grid_layout.addWidget(widget, row, col)

    # -------------------------
    # ADD CAMERA
    # -------------------------
    def add_camera_menu(self):
        busy_cameras = [stream.source for stream in self.streams if isinstance(stream.source, int)]
        cams = list_cameras(exclude_indices=busy_cameras)

        if not cams:
            QMessageBox.warning(self, "Erro", "Nenhuma câmera encontrada")
            return

        # Transforma lista em strings amigáveis
        items = [f"Câmera {i}" for i in cams]

        item, ok = QInputDialog.getItem(
            self,
            "Selecionar câmera",
            "Escolha uma câmera:",
            items,
            0,
            False
        )

        if ok and item:
            index = items.index(item)
            cam_id = cams[index]
            stream_name = f"Câmera {cam_id}"

            self.add_stream(cam_id, stream_name)

    # -------------------------
    # ADD VIDEO
    # -------------------------
    def add_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Selecionar vídeo", "", "Videos (*.mp4 *.avi *.mkv)"
        )

        if path:
            # Extrai nome do arquivo sem extensão
            stream_name = os.path.basename(path).split('.')[0]
            self.add_stream(path, stream_name)

    # -------------------------
    # STREAM CORE
    # -------------------------
    def add_stream(self, source, source_name="Stream"):
        """Adicionar uma nova stream (câmera ou vídeo)"""
        # Criar widget com metadados
        widget = StreamWidget(source, source_name)
        self.streams.append(widget)

        # Criar worker
        worker = VideoWorker(source)
        
        # Conectar worker ao widget
        widget.set_worker(worker)
        
        # Conectar sinal de fechar
        widget.close_signal.connect(lambda: self.remove_stream(widget))
        
        # Iniciar worker
        worker.start()
        self.workers.append(worker)

        # Agendar relayout (deferred)
        self.relayout_streams()

    def remove_stream(self, widget):
        """Remover uma stream"""
        if widget in self.streams:
            idx = self.streams.index(widget)
            self.streams.remove(widget)
            
            if idx < len(self.workers):
                worker = self.workers.pop(idx)
                worker.stop()
            
            widget.cleanup()
            widget.setParent(None)
            
            # Reorganizar layout
            self.relayout_streams()

    # -------------------------
    # TEMA
    # -------------------------
    def toggle_theme(self):
        pass

    def apply_theme(self, dark=True):
        """Aplicar tema à interface"""
        if dark:
            QApplication.instance().setStyleSheet(DARK_THEME)
            # enfatiza top bar em modo escuro
            try:
                self.top_bar_widget.setStyleSheet("background-color: #2b2b2b; border-radius: 12px;")
            except Exception:
                pass
            self.dark_mode = True
        else:
            QApplication.instance().setStyleSheet(LIGHT_THEME)
            try:
                self.top_bar_widget.setStyleSheet("background-color: #e5e5e5; border-radius: 12px;")
            except Exception:
                pass
            self.dark_mode = False
    
    def open_onnx_config(self):
        """Abrir diálogo de configuração de modelos ONNX."""
        dialog = OnnxConfigDialog(self)
        dialog.exec()