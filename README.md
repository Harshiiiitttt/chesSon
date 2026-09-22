# ♟️ chesSon: Deep Learning Chess Engine

`chesSon` is a chess bot engine trained on over 100k Lichess broadcast tournament games using Supervised Learning.

## Architecture

* **Board Representation**: Canonical $8 \times 8 \times 14$ float32 tensor (12 piece planes, 1 castling rights plane, 1 en-passant plane) oriented to the player to move.
* **Action Space**: 4,168 discrete actions ($64 \times 64$ move matrix + promotions/underpromotions).
* **Neural Network (`ChesSonNet`)**: Dual-head ResNet (6 residual blocks, 128 channels).
  * **Policy Head**: Logits over 4,168 actions.
  * **Value Head**: Position evaluation scalar in $[-1, 1]$ (Tanh).
* **Search Engine**: Alpha-Beta Negamax with Policy-based move ordering and neural value evaluation.
* **Protocol**: Universal Chess Interface (UCI) compliant.

---

## Folder Structure

```
chesSon/
├── chesson/
│   ├── board_encoding.py    # Canonical 8x8x14 tensor & 4168 move index encoder
│   ├── model.py             # Dual-head ResNet (Policy & Value)
│   ├── search.py            # Negamax with Alpha-Beta pruning & Policy ordering
│   ├── uci.py               # Standard Universal Chess Interface loop
│   └── engine.py            # Main entry point (CLI, UCI, Diagnostics)
├── kaggle/
│   ├── chesSon_training_kaggle.ipynb # Ready-to-upload Kaggle notebook
│   └── train_kaggle.py               # Standalone training script
├── tests/
│   └── test_engine.py       # Unit tests
├── weights/
│   └── (Drop chesson_weights.pt here after training on Kaggle)
├── requirements.txt
└── README.md
```

---

## How to Train on Kaggle

1. **Upload Dataset**: On Kaggle, create a new Dataset and upload your `.pgn.zst` broadcast file.
2. **Create Notebook**:
   * Click **New Notebook** on Kaggle.
   * Go to **File -> Import Notebook** and upload [`kaggle/chesSon_training_kaggle.ipynb`](kaggle/chesSon_training_kaggle.ipynb).
   * In the right sidebar under **Settings**:
     * Accelerator: Set to **GPU T4 x2** or **GPU P100**.
     * Internet: **On**.
3. **Run All Cells**:
   * The notebook streams data on the fly from the `.pgn.zst` file, avoiding Kaggle disk space limits.
   * Trains using FP16 mixed precision and AdamW.
4. **Download Weights**:
   * Once training completes, download `chesson_weights.pt` from the **Output** tab.
   * Save it into your local `chesSon/weights/` folder.

---

## How to Run chesSon Locally

### 1. Interactive CLI Mode (Play in Terminal)
```powershell
python -m chesson.engine --play --color white --weights weights/chesson_weights.pt --depth 2
```

### 2. Standard UCI Mode (for Arena, CuteChess, or Lichess-bot)
```powershell
python -m chesson.engine --weights weights/chesson_weights.pt
```

### 3. Run Self-Diagnostics
```powershell
python -m chesson.engine --test
```

### 4. Run Unit Tests
```powershell
python tests/test_engine.py
```
