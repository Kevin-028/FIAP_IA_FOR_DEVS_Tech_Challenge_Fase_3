"""
collect_data.py
---------------
Coleta landmarks da mão via webcam e salva em data/landmarks.csv.

Como usar:
  python collect_data.py

Pressione o número (0-9) para iniciar a gravação daquele gesto.
Pressione 'Q' para sair.
"""

import csv
import os
import time

import cv2
import mediapipe as mp

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "landmarks.csv")
SAMPLES_PER_DIGIT = 200   # frames coletados por dígito por sessão
COUNTDOWN = 3             # segundos de contagem antes de gravar

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def extract_landmarks(hand_landmarks):
    """Normaliza os 21 landmarks em relação ao pulso (landmark 0)."""
    pts = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
    ox, oy, oz = pts[0]
    flat = []
    for x, y, z in pts:
        flat.extend([x - ox, y - oy, z - oz])
    return flat


def main():
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

    file_exists = os.path.isfile(DATA_PATH)
    csv_file = open(DATA_PATH, "a", newline="")
    writer = csv.writer(csv_file)
    if not file_exists:
        header = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
        writer.writerow(["label"] + header)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as hands:

        collecting = False
        current_digit = None
        count = 0
        start_time = None

        print("=" * 50)
        print("  COLETA DE DADOS — GESTOS 0 a 9")
        print("  Pressione 0-9 para gravar um dígito")
        print("  Pressione Q para sair")
        print("=" * 50)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hl in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame, hl,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

                    if collecting and count < SAMPLES_PER_DIGIT:
                        elapsed = time.time() - start_time
                        if elapsed >= COUNTDOWN:
                            landmarks = extract_landmarks(hl)
                            writer.writerow([current_digit] + landmarks)
                            csv_file.flush()
                            count += 1

            # HUD
            if collecting:
                elapsed = time.time() - start_time
                if elapsed < COUNTDOWN:
                    msg = f"Preparando digito {current_digit}... {COUNTDOWN - int(elapsed)}s"
                    cv2.putText(frame, msg, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, (0, 200, 255), 2)
                else:
                    remaining = SAMPLES_PER_DIGIT - count
                    msg = f"Gravando {current_digit}: {count}/{SAMPLES_PER_DIGIT}"
                    cv2.putText(frame, msg, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, (0, 255, 0), 2)
                    cv2.rectangle(frame, (20, 65),
                                  (20 + int(580 * count / SAMPLES_PER_DIGIT), 85),
                                  (0, 255, 0), -1)

                    if count >= SAMPLES_PER_DIGIT:
                        print(f"  ✓ Dígito {current_digit} concluído!")
                        collecting = False
                        count = 0
            else:
                cv2.putText(frame, "Pressione 0-9 para gravar | Q para sair",
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (200, 200, 200), 1)

            cv2.imshow("Coleta de Dados — Gestos", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == ord("Q"):
                break
            elif ord("0") <= key <= ord("9") and not collecting:
                current_digit = key - ord("0")
                collecting = True
                count = 0
                start_time = time.time()
                print(f"  → Iniciando coleta do dígito {current_digit}...")

    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()
    print("\nColeta encerrada. Arquivo salvo em:", DATA_PATH)


if __name__ == "__main__":
    main()
