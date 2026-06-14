import cv2
import numpy as np
import openvino as ov
import math
import time

# =========================================================
# 1. CONFIGURAÇÕES GERAIS E ARQUIVOS CONVERTIDOS (.XML)
# =========================================================
VIDEO_PATH = "videos/1000lh.mp4"

UNET_VISOR_XML = "pesos/visor_openvino.xml"
YOLO_PONTEIRO_XML = "pesos/ponteiro_openvino.xml"
OCR_XML = "pesos/ocr_openvino.xml"
YOLO_POSE_XML = "pesos/posicao_openvino.xml"

SEG_SIZE = 512
YOLO_SIZE = 640
YOLO_POSE_SIZE = 320  # OTIMIZAÇÃO CRUCIAL: Mantém a Pose leve para a Iris Xe

THRESHOLD_UNET = 0.5
CONF_THRESHOLD_YOLO = 0.4
CONF_THRESHOLD_POSE = 0.25
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

print("Inicializando OpenVINO Core com Otimizações de Hardware Intel...")
core = ov.Core()
core.set_property({'CACHE_DIR': './ov_cache_dir'})

# Voltando para o modo síncrono focado em latência pura e direta
config_gpu = {"PERFORMANCE_HINT": "LATENCY"}
config_cpu = {"INFERENCE_NUM_THREADS": "4"}

print("Compilando modelos nativamente para Intel Iris Xe (GPU)...")
t_start_comp = time.perf_counter()
compiled_visor = core.compile_model(core.read_model(UNET_VISOR_XML), "GPU", config_gpu)
compiled_yolo = core.compile_model(core.read_model(YOLO_PONTEIRO_XML), "GPU", config_gpu)
compiled_pose = core.compile_model(core.read_model(YOLO_POSE_XML), "GPU", config_gpu)

print("Compilando OCR para Intel Core i7 (CPU)...")
compiled_ocr = core.compile_model(core.read_model(OCR_XML), "CPU", config_cpu)
print(f"-> Tempo total de compilação: {(time.perf_counter() - t_start_comp):.2f} segundos.")

infer_visor = compiled_visor.create_infer_request()
infer_yolo = compiled_yolo.create_infer_request()
infer_pose = compiled_pose.create_infer_request()
infer_ocr = compiled_ocr.create_infer_request()

# WARM-UP MANTIDO: Evita o engasgo do Frame 1 de forma limpa e síncrona
print("[Warm-up] Aquecendo os motores da GPU com execução direta...")
infer_visor.infer([np.ascontiguousarray(np.zeros((1, 3, SEG_SIZE, SEG_SIZE), dtype=np.float32))])
infer_yolo.infer([np.ascontiguousarray(np.zeros((1, 3, YOLO_SIZE, YOLO_SIZE), dtype=np.float32))])
infer_pose.infer([np.ascontiguousarray(np.zeros((1, 3, YOLO_POSE_SIZE, YOLO_POSE_SIZE), dtype=np.float32))])
infer_ocr.infer([np.ascontiguousarray(np.zeros((1, 1, 32, 128), dtype=np.float32))])
print("[Warm-up Concluído] Estabilidade garantida.")


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
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


def processar_video():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {VIDEO_PATH}")
        return

    calibrado = False
    Matriz_H = None
    dimensoes_finais = None
    rotation_step = None

    bbox_visor_estatico = None
    caixas_ponteiros_estaticas = []

    contador_frames = 0
    FRAME_SKIP = 3

    texto_ocr_cache = "Carregando..."
    p_superior_cache = 0.0
    p_inferior_cache = 0.0
    linhas_ponteiros_hud = []

    fps_atual = 0.0
    acumulador_tempo_fps = 0.0
    contador_fps = 0
    t_anterior = time.perf_counter()

    NOME_JANELA = "Monitoramento Otimizado Sincrono"
    cv2.namedWindow(NOME_JANELA, cv2.WINDOW_NORMAL)

    while True:
        t_inicio_frame = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            break

        contador_frames += 1
        h_orig, w_orig = frame.shape[:2]

        # -----------------------------------------------------
        # CALIBRAÇÃO GEOMÉTRICA (APENAS FRAME 1)
        # -----------------------------------------------------
        if not calibrado:
            t_inicio_calib = time.perf_counter()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            seg_inp = cv2.resize(frame_rgb, (SEG_SIZE, SEG_SIZE)).astype(np.float32) / 255.0
            seg_inp = np.transpose(seg_inp, (2, 0, 1))[None, ...]
            seg_inp = np.ascontiguousarray(seg_inp)

            # Voltou para infer síncrono (Mais estável para processadores Intel U)
            pred_seg = infer_visor.infer([seg_inp])[0]
            mask_prob = 1 / (1 + np.exp(-pred_seg))
            mask_v = (np.squeeze(mask_prob) > THRESHOLD_UNET).astype(np.uint8)
            mask_v = cv2.resize(mask_v, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

            contours_v, _ = cv2.findContours(mask_v, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours_v) == 0: return
            maior_cnt = max(contours_v, key=cv2.contourArea)
            x_b, y_b, w_b, h_b = cv2.boundingRect(maior_cnt)
            centro_x_visor = x_b + (w_b / 2)
            centro_y_visor = y_b + (h_b / 2)

            inp_img, ratio, dw, dh = letterbox(frame_rgb, new_shape=(YOLO_SIZE, YOLO_SIZE))
            yolo_inp = np.transpose(inp_img.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]
            yolo_inp = np.ascontiguousarray(yolo_inp)

            saida_bruta_yolo = infer_yolo.infer([yolo_inp])[0]

            caixas_p = []
            for pred in saida_bruta_yolo[0]:
                if pred[4] < CONF_THRESHOLD_YOLO: continue
                caixas_p.append([
                    int(max(0, (pred[0] - dw) / ratio)),
                    int(max(0, (pred[1] - dh) / ratio)),
                    int(min(w_orig, (pred[2] - dw) / ratio)),
                    int(min(h_orig, (pred[3] - dh) / ratio))
                ])

            if len(caixas_p) == 0: return
            centro_x_ponteiros = np.mean([(b[0] + b[2]) / 2 for b in caixas_p])
            centro_y_ponteiros = np.mean([(b[1] + b[3]) / 2 for b in caixas_p])

            frame_calib = frame.copy()
            mask_calib = mask_v.copy()
            if h_b > w_b:
                if centro_x_ponteiros > centro_x_visor:
                    frame_calib = cv2.rotate(frame_calib, cv2.ROTATE_90_CLOCKWISE)
                    mask_calib = cv2.rotate(mask_calib, cv2.ROTATE_90_CLOCKWISE)
                    rotation_step = cv2.ROTATE_90_CLOCKWISE
                elif centro_x_ponteiros < centro_x_visor:
                    frame_calib = cv2.rotate(frame_calib, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    mask_calib = cv2.rotate(mask_calib, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    rotation_step = cv2.ROTATE_90_COUNTERCLOCKWISE
            else:
                if centro_y_visor > centro_y_ponteiros:
                    frame_calib = cv2.rotate(frame_calib, cv2.ROTATE_180)
                    mask_calib = cv2.rotate(mask_calib, cv2.ROTATE_180)
                    rotation_step = cv2.ROTATE_180

            h_c, w_c = frame_calib.shape[:2]

            contours_c, _ = cv2.findContours(mask_calib, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
                maior_cnt_c = max(contours_c, key=cv2.contourArea)

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
            margem_w, margem_h = int(largura_visor * 0.8), int(altura_visor * 3.0)

            Matriz_H = cv2.getPerspectiveTransform(pts_origem, pts_destino := np.array([
                [margem_w, margem_h], [margem_w + largura_visor, margem_h],
                [margem_w + largura_visor, margem_h + altura_visor], [margem_w, margem_h + altura_visor]
            ], dtype="float32"))
            dimensoes_finais = (largura_visor + margem_w * 2, altura_visor + margem_h * 2)

            frame_piloto_est = cv2.warpPerspective(frame_calib, Matriz_H, dimensoes_finais)
            h_est, w_est = frame_piloto_est.shape[:2]
            img_rgb_est = cv2.cvtColor(frame_piloto_est, cv2.COLOR_BGR2RGB)

            seg_inp_est = cv2.resize(img_rgb_est, (SEG_SIZE, SEG_SIZE)).astype(np.float32) / 255.0
            seg_inp_est = np.transpose(seg_inp_est, (2, 0, 1))[None, ...]
            seg_inp_est = np.ascontiguousarray(seg_inp_est)

            pred_seg_est = infer_visor.infer([seg_inp_est])[0]
            mask_v_est = (np.squeeze(1 / (1 + np.exp(-pred_seg_est))) > THRESHOLD_UNET).astype(np.uint8)
            mask_v_est = cv2.resize(mask_v_est, (w_est, h_est), interpolation=cv2.INTER_NEAREST)
            contours_est, _ = cv2.findContours(mask_v_est, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours_est) > 0:
                bbox_visor_estatico = cv2.boundingRect(max(contours_est, key=cv2.contourArea))

            inp_img_est, ratio_est, dw_est, dh_est = letterbox(img_rgb_est, new_shape=(YOLO_SIZE, YOLO_SIZE))
            yolo_inp_est = np.transpose(inp_img_est.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]
            yolo_inp_est = np.ascontiguousarray(yolo_inp_est)
            saida_bruta_yolo_est = infer_yolo.infer([yolo_inp_est])[0]

            ponteiros_validos = []
            for pred in saida_bruta_yolo_est[0]:
                if pred[4] < CONF_THRESHOLD_YOLO: continue
                ponteiros_validos.append([
                    int(max(0, (pred[0] - dw_est) / ratio_est)), int(max(0, (pred[1] - dh_est) / ratio_est)),
                    int(min(w_est, (pred[2] - dw_est) / ratio_est)), int(min(h_est, (pred[3] - dh_est) / ratio_est))
                ])
            caixas_ponteiros_estaticas = sorted(ponteiros_validos, key=lambda p: p[1])
            calibrado = True
            print(f"[Calibração Concluída]: {(time.perf_counter() - t_inicio_calib) * 1000:.1f}ms")

        # -----------------------------------------------------
        # PROCESSAMENTO EM TEMPO REAL
        # -----------------------------------------------------
        t_inicio_warp = time.perf_counter()
        if rotation_step is not None:
            frame = cv2.rotate(frame, rotation_step)

        frame_estabilizado = cv2.warpPerspective(frame, Matriz_H, dimensoes_finais)
        tempo_warp = (time.perf_counter() - t_inicio_warp) * 1000.0

        tempo_ocr_ms = 0.0
        tempo_pose_ms = 0.0
        veio_do_skip = True

        if contador_frames % FRAME_SKIP == 0 or contador_frames == 1:
            veio_do_skip = False
            linhas_ponteiros_hud.clear()
            valores_analogicos = []

            # 1. OCR Síncrono em CPU
            t_inicio_ocr = time.perf_counter()
            if bbox_visor_estatico is not None:
                x_v, y_v, ww_v, hh_v = bbox_visor_estatico
                crop_visor = frame_estabilizado[y_v:y_v + hh_v, x_v:x_v + ww_v]
                if crop_visor.size > 0:
                    crop_gray = cv2.cvtColor(crop_visor, cv2.COLOR_BGR2GRAY)
                    crop_resized = cv2.resize(crop_gray, (128, 32), interpolation=cv2.INTER_LINEAR)
                    tensor_ocr = np.expand_dims(crop_resized.astype(np.float32) / 255.0, axis=(0, 1))
                    tensor_ocr = np.ascontiguousarray(tensor_ocr)

                    pred_ocr = infer_ocr.infer([tensor_ocr])[0]
                    pred_argmax = np.argmax(pred_ocr, axis=2).flatten()

                    texto_ocr_cache = ""
                    last = -1
                    for p in pred_argmax:
                        if p != last and p != 0 and (p - 1) < len(CHARS):
                            texto_ocr_cache += CHARS[p - 1]
                        last = p
                    if not texto_ocr_cache: texto_ocr_cache = "..."
            tempo_ocr_ms = (time.perf_counter() - t_inicio_ocr) * 1000.0

            # 2. YOLO Pose Síncrono em GPU
            t_inicio_pose = time.perf_counter()
            for idx, ponteiro in enumerate(caixas_ponteiros_estaticas):
                x1, y1, x2, y2 = ponteiro
                crop_p = frame_estabilizado[y1:y2, x1:x2]
                if crop_p.size == 0: continue

                crop_rgb = cv2.cvtColor(crop_p, cv2.COLOR_BGR2RGB)
                inp_pose, ratio_p, dw_p, dh_p = letterbox(crop_rgb, new_shape=(YOLO_POSE_SIZE, YOLO_POSE_SIZE))
                inp_p_tensor = np.transpose(inp_pose.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]
                inp_p_tensor = np.ascontiguousarray(inp_p_tensor)

                preds_pose = infer_pose.infer([inp_p_tensor])[0]
                preds_pose = np.transpose(preds_pose[0])

                melhor_score = 0.0
                melhor_pred = None
                for pred in preds_pose:
                    if pred[4] > melhor_score and pred[4] >= CONF_THRESHOLD_POSE:
                        melhor_score = pred[4]
                        melhor_pred = pred

                if melhor_pred is not None:
                    kpts = melhor_pred[5:]
                    if kpts[2] > 0.35 and kpts[5] > 0.35:
                        eixo_x = (kpts[0] - dw_p) / ratio_p
                        eixo_y = (kpts[1] - dh_p) / ratio_p
                        ponta_x = (kpts[3] - dw_p) / ratio_p
                        ponta_y = (kpts[4] - dh_p) / ratio_p

                        angulo_ajustado = math.degrees(math.atan2(ponta_y - eixo_y, ponta_x - eixo_x)) + 90.0
                        if angulo_ajustado < 0: angulo_ajustado += 360.0
                        valores_analogicos.append(round((angulo_ajustado % 360.0) / 36.0, 1))

                        linhas_ponteiros_hud.append({
                            "eixo": (int(x1 + eixo_x), int(y1 + eixo_y)),
                            "ponta": (int(x1 + ponta_x), int(y1 + ponta_y))
                        })
                    else:
                        values = 0.0
                else:
                    values = 0.0

            p_superior_cache = valores_analogicos[0] if len(valores_analogicos) > 0 else 0.0
            p_inferior_cache = valores_analogicos[1] if len(valores_analogicos) > 1 else 0.0
            tempo_pose_ms = (time.perf_counter() - t_inicio_pose) * 1000.0

        # --- HUD ---
        for idx, ponteiro in enumerate(caixas_ponteiros_estaticas, start=1):
            cv2.rectangle(frame_estabilizado, (ponteiro[0], ponteiro[1]), (ponteiro[2], ponteiro[3]), (255, 0, 0), 2)
        for pts_hud in linhas_ponteiros_hud:
            cv2.line(frame_estabilizado, pts_hud["eixo"], pts_hud["ponta"], (0, 255, 255), 2)
        if bbox_visor_estatico is not None:
            cv2.rectangle(frame_estabilizado, (bbox_visor_estatico[0], bbox_visor_estatico[1]),
                          (bbox_visor_estatico[0] + bbox_visor_estatico[2],
                           bbox_visor_estatico[1] + bbox_visor_estatico[3]), (0, 255, 0), 2)

        t_fim_frame = time.perf_counter()
        tempo_total_frame_ms = (t_fim_frame - t_inicio_frame) * 1000.0

        acumulador_tempo_fps += (t_fim_frame - t_anterior)
        t_anterior = t_fim_frame
        contador_fps += 1
        if contador_fps >= 10:
            fps_atual = contador_fps / acumulador_tempo_fps
            contador_fps = 0
            acumulador_tempo_fps = 0.0

        tipo_frame = "INFERENCIA" if not veio_do_skip else "SKIP (CACHE)"
        print(
            f"Frame {contador_frames:04d} [{tipo_frame}] -> Total: {tempo_total_frame_ms:.1f}ms | Warp: {tempo_warp:.1f}ms | OCR: {tempo_ocr_ms:.1f}ms | Pose (x2): {tempo_pose_ms:.1f}ms | FPS: {fps_atual:.1f}")

        cv2.rectangle(frame_estabilizado, (0, 0), (frame_estabilizado.shape[1], 45), (0, 0, 0), -1)
        telemetria = f"FPS: {fps_atual:.1f} ({tempo_total_frame_ms:.1f}ms) | OCR: {texto_ocr_cache} | P. Sup: {p_superior_cache:.1f} | P. Inf: {p_inferior_cache:.1f}"
        cv2.putText(frame_estabilizado, telemetria, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
                    cv2.LINE_AA)

        cv2.imshow(NOME_JANELA, frame_estabilizado)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    processar_video()