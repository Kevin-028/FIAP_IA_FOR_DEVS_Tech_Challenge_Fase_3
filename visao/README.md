# Visão Computacional — Reconhecimento de Gestos (1–5)

Módulo de visão computacional que identifica **números de 1 a 5** mostrados com a mão via webcam, usando **MediaPipe Hands** + **Random Forest**.

> Integrado à aplicação web Flask em `/visao`. Os scripts desta pasta oferecem uma alternativa de uso via terminal.

---

## Como funciona

```
Webcam → MediaPipe (21 landmarks 3D) → normalização pelo pulso → Random Forest → Dígito (1–5)
```

O MediaPipe extrai 21 pontos 3D da mão. Os landmarks são normalizados subtraindo a posição do pulso (ponto 0), gerando 63 features que alimentam o classificador.

> **Por que apenas 1–5?**  
> Cada número de 1 a 5 pode ser representado de forma inequívoca com **uma única mão** (contagem de dedos erguidos). Números de 6 a 9 não possuem representação universal com uma mão — dependem de convenções culturais ou requerem as duas mãos.

---

## Uso via Web (recomendado)

Com a aplicação Flask rodando (`F5` no VS Code ou `cd web && python app.py`):

| Rota | Ação |
|---|---|
| `/visao/coletar` | Coleta amostras via webcam no navegador |
| `/visao/treinar` | Treina o modelo e exibe métricas |
| `/visao` | Predição em tempo real (suporta 1 ou 2 mãos) |

As bibliotecas MediaPipe ficam em `web/static/mediapipe/` — **sem dependência de CDN**.

---

## Uso via Terminal

### 1. Coletar dados de treino

```bash
python visao/collect_data.py
```

- Pressione `1`–`5` para iniciar a gravação de cada dígito (200 frames por dígito)
- Varie ângulos e distâncias para um modelo mais robusto
- Pressione `Q` para sair
- Dados salvos em `visao/data/landmarks.csv`

### 2. Treinar o modelo

```bash
python visao/train_model.py
```

- Treina Random Forest com validação cruzada (5-fold)
- Exibe relatório de classificação e matriz de confusão
- Salva modelo em `visao/modelo/modelo_visao.pkl`

### 3. Predição em tempo real

```bash
python visao/predict_live.py
```

- Abre a webcam com OpenCV
- Mostra o dígito reconhecido e a confiança em tempo real
- Pressione `Q` para sair

---

## Estrutura

```
visao/
├── collect_data.py           # Coleta landmarks via webcam (terminal)
├── train_model.py            # Treina e avalia o classificador
├── predict_live.py           # Reconhecimento em tempo real (OpenCV)
├── data/
│   └── landmarks.csv         # Dataset coletado (gerado em runtime)
└── modelo/
    ├── modelo_visao.pkl       # Modelo serializado (gerado em runtime)
    └── confusion_matrix.png   # Matriz de confusão (gerado em runtime)
```

---

## Dependências

Instaladas via `requirements.txt` na raiz do projeto:

```
mediapipe
opencv-python
scikit-learn
joblib
numpy
pandas
matplotlib
seaborn
```

