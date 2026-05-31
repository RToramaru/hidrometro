"""
Arquivo centralizado de constantes e configurações.
Gerencia paths dos modelos ONNX, flags de habilitação e configurações de performance.
"""

import os
import sys

# ============================================================
# PATHS DOS MODELOS ONNX
# ============================================================
MODELS_PATHS = {
    "visor": "pesos/visor.onnx",
    "ponteiros": "pesos/ponteiro.onnx",
    "ocr": "pesos/ocr.onnx",
    "pose": "pesos/posicao.onnx",
}

# ============================================================
# FLAGS DE HABILITAÇÃO DOS MODELOS
# ============================================================
CONFIG_MODELS_ENABLED = {
    "visor": True,
    "ponteiros": True,
    "ocr": True,
    "pose": True,
}

# ============================================================
# CONFIGURAÇÃO DE PERFORMANCE
# ============================================================
# Processar a cada N frames (para não impactar FPS)
# Quanto maior, mais rápido, mas menos frequente o processamento
PROCESS_INTERVAL = 5

# Habilitar detecção automática de rotação no primeiro frame
AUTO_ROTATION = True

# ============================================================
# CONFIGURAÇÕES ONNX RUNTIME
# ============================================================
# Providers priorizados: GPU se disponível, fallback CPU
ONNX_PROVIDERS = ['CUDAExecutionProvider', 'CPUExecutionProvider']
ONNX_PROVIDERS_CPU_ONLY = ['CPUExecutionProvider']

# ============================================================
# CONFIGURAÇÕES DOS MODELOS
# ============================================================
SEG_SIZE = 512
YOLO_SIZE = 640

# Thresholds
THRESHOLD_UNET = 0.5
CONF_THRESHOLD_YOLO = 0.4
CONF_THRESHOLD_POSE = 0.25

# OCR
OCR_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# ============================================================
# VALIDAÇÃO E INICIALIZAÇÃO
# ============================================================

def validate_models():
    """
    Valida se todos os modelos estão presentes.
    Se algum arquivo não existir, desabilita o modelo automaticamente.
    Retorna dicionário com status de cada modelo.
    """
    status = {}
    
    for model_name, path in MODELS_PATHS.items():
        if os.path.exists(path):
            status[model_name] = {
                "available": True,
                "path": os.path.abspath(path),
            }
            print(f"✓ Modelo '{model_name}' encontrado: {path}")
        else:
            status[model_name] = {
                "available": False,
                "path": path,
            }
            CONFIG_MODELS_ENABLED[model_name] = False
            print(f"✗ AVISO: Modelo '{model_name}' NÃO ENCONTRADO em {path}")
            print(f"  → Desabilitando '{model_name}' automaticamente")
    
    return status


def is_model_enabled(model_name: str) -> bool:
    """Retorna se um modelo específico está habilitado."""
    return CONFIG_MODELS_ENABLED.get(model_name, False)


def enable_model(model_name: str):
    """Habilita um modelo específico."""
    if model_name in CONFIG_MODELS_ENABLED:
        CONFIG_MODELS_ENABLED[model_name] = True


def disable_model(model_name: str):
    """Desabilita um modelo específico."""
    if model_name in CONFIG_MODELS_ENABLED:
        CONFIG_MODELS_ENABLED[model_name] = False


def is_any_model_enabled() -> bool:
    """Retorna se qualquer modelo está habilitado."""
    return any(CONFIG_MODELS_ENABLED.values())


def get_enabled_models() -> list:
    """Retorna lista de modelos habilitados."""
    return [name for name, enabled in CONFIG_MODELS_ENABLED.items() if enabled]


# Validar modelos ao importar o arquivo
print("\n" + "="*60)
print("Validando disponibilidade dos modelos ONNX...")
print("="*60)
MODELS_STATUS = validate_models()
print("="*60 + "\n")
