"""
predict_live.py
---------------
Reconhecimento de gestos (0-9) em tempo real via webcam.

Como usar:
  python predict_live.py

Pressione Q para sair.
"""

import os
import time
import collections

import cv2
import joblib
import mediapipe as mp
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "modelo", "modelo_visao.pkl")

# Cores FIAP
COR_FUNDO   = (26, 26, 26)      # #1A1A1A
COR_DESTAQUE = (64, 51, 239)    # #EF3340 em BGR
COR_BRANCO  = (255, 255, 255)
COR_VERDE   = (86, 214, 71)
COR_CINZA   = (160, 160, 160)

mp_hands  = mp.solutions.hands
mp_draw   = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def extract_landmarks(hand_landmarks):
    """Normaliza os 21 landmarks em relação ao pulso (landmark 0)."""
    pts = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
    ox, oy, oz = pts[0]
    flat = []
    for x, y, z in pts:
        flat.extend([x - ox, y - oy, z - oz])
    return flat


def draw_rounded_rect(img, x1, y1, x2, y2, r, color, thickness=-1):
    """Desenha retângulo com cantos arredondados."""
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, thickness)
    for cx, cy in [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]:
        cv2.ellipse(img, (cx, cy), (r, r), 0, 0, 0, color, thickness)
        cv2.ellipse(img, (cx, cy), (r, r), 90, 0, 0, color, thickness)
        cv2.ellipse(img, (cx, cy), (r, r), 180, 0, 0, color, thickness)
        cv2.ellipse(img, (cx, cy), (r, r), 270, 0, 0, color, thickness)


def main():
    if not os.path.isfile(MODEL_PATH):
        print("ERRO: modelo não encontrado.")
        print("Execute train_model.py primeiro.")
        return

    payload = joblib.load(MODEL_PATH)
    model   = payload["modelo"]
    le      = payload["label_encoder"]
    metricas = payload.get("metricas", {})

    print("Modelo carregado.")
    print(f"  Acurácia (CV): {metricas.get('cv_accuracy_mean', 0):.2%}")
    print("  Pressione Q para sair.\n")

    # Suavização de predições (últimos N frames)
    SMOOTH_N  = 10
    history   = collections.deque(maxlen=SMOOTH_N)
    fps_times = collections.deque(maxlen=30)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as hands:

        while cap.isOpened():
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            detected = False
            pred_digit = None
            confidence = 0.0

            if results.multi_hand_landmarks:
                for hl in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame, hl,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )
                    lm = extract_landmarks(hl)
                    proba = model.predict_proba([lm])[0]
                    idx = int(np.argmax(proba))
                    confidence = float(proba[idx])
                    pred_digit = int(le.inverse_transform([idx])[0])
                    history.append(pred_digit)
                    detected = True
                    break  # só 1 mão

            # Dígito suavizado (moda dos últimos frames)
            smooth_digit = None
            if history:
                smooth_digit = max(set(history), key=history.count)

            # ── Painel lateral direito ──────────────────────────────────
            panel_w = 160
            panel_x = w - panel_w
            overlay = frame.copy()
            cv2.rectangle(overlay, (panel_x, 0), (w, h), COR_FUNDO, -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

            # Título no painel
            cv2.putText(frame, "GESTO", (panel_x + 30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COR_CINZA, 1)

            if smooth_digit is not None and detected:
                # Número grande
                digit_str = str(smooth_digit)
                font_scale = 5.0
                thickness  = 8
                (tw, th), _ = cv2.getTextSize(digit_str, cv2.FONT_HERSHEY_SIMPLEX,
                                               font_scale, thickness)
                tx = panel_x + (panel_w - tw) // 2
                cv2.putText(frame, digit_str, (tx, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            COR_DESTAQUE, thickness)

                # Barra de confiança
                bar_x1 = panel_x + 10
                bar_x2 = w - 10
                bar_y  = 185
                bar_h  = 14
                cv2.rectangle(frame, (bar_x1, bar_y), (bar_x2, bar_y + bar_h),
                              COR_CINZA, 1)
                filled = int((bar_x2 - bar_x1 - 2) * confidence)
                cv2.rectangle(frame, (bar_x1 + 1, bar_y + 1),
                              (bar_x1 + 1 + filled, bar_y + bar_h - 1),
                              COR_VERDE, -1)
                cv2.putText(frame, f"{confidence:.0%}", (bar_x1, bar_y + bar_h + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, COR_BRANCO, 1)

            else:
                cv2.putText(frame, "---", (panel_x + 38, 155),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.5, COR_CINZA, 4)
                cv2.putText(frame, "Mostre a mao", (panel_x + 5, 195),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, COR_CINZA, 1)

            # FPS
            fps_times.append(time.time() - t0)
            fps = 1.0 / (sum(fps_times) / len(fps_times))
            cv2.putText(frame, f"FPS: {fps:.0f}", (panel_x + 10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COR_CINZA, 1)

            # Legenda inferior
            cv2.putText(frame, "Q = sair", (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COR_CINZA, 1)

            cv2.imshow("Reconhecimento de Gestos — 0 a 9", frame)

            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
