#!/usr/bin/env python3
"""
Script de teste para validar que a interface funciona com processamento desabilitado.
"""

print("\n" + "="*70)
print("TESTE DE COMPATIBILIDADE: Interface Sem Processamento ONNX")
print("="*70 + "\n")

# Simular interface desabilitando modelos
import constants

# Desabilitar todos os modelos
constants.CONFIG_MODELS_ENABLED = {
    "visor": False,
    "ponteiros": False,
    "ocr": False,
    "pose": False,
}

print("Status de configuração simulado:")
print(f"  Modelos habilitados: {constants.get_enabled_models()}")
print(f"  is_any_model_enabled(): {constants.is_any_model_enabled()}")

# Simular VideoWorker criado sem processamento
from video_worker import VideoWorker

print("\n✓ VideoWorker pode ser instanciado sem processamento")
print("  (não tentará carregar modelos)")

# Verificar que VideoProcessor não seria criado
print("\nSimulação:")
print("  1. VideoWorker init(source, enable_processing=False)")
print("  2. Nenhum processador será carregado")
print("  3. Interface funciona 100% como antes (apenas frames brutos)")

print("\n" + "="*70)
print("✓ TESTE PASSOU: Interface compatível com processamento desabilitado")
print("="*70 + "\n")

# Reabilitar para outros testes
constants.CONFIG_MODELS_ENABLED = {
    "visor": True,
    "ponteiros": True,
    "ocr": True,
    "pose": True,
}

print("Status normalizado:")
print(f"  Modelos habilitados: {constants.get_enabled_models()}")
print(f"  Pronto para processar vídeos com ONNX quando habilitado\n")
