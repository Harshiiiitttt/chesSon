"""Self-contained training script for chesSon on Kaggle — v2 (100K+ Games Edition).

This script:
1. Streams compressed .pgn.zst games on-the-fly (zero disk exhaustion).
2. Also supports plain .pgn files and the popular Kaggle CSV dataset format.
3. Supports multiple input files — mix and match formats for maximum data.
4. Fine-tuning mode: loads existing weights and continues training.
5. Filters games (moves > min_moves, valid outcomes, learns from winner's moves).
6. Encodes positions into 8x8x14 canonical tensors and 4168-action indices.
7. Trains ChesSonNet using mixed precision (FP16) on Kaggle GPU.
8. Saves the final weights as 'chesson_weights.pt'.

Supported Kaggle datasets:
  - Any .pgn.zst (e.g., Lichess monthly dumps)
  - Any .pgn (e.g., FICS, ICC, ChessBase exports)
  - 'games.csv' from https://www.kaggle.com/datasets/datasnaek/chess
    (columns: moves, winner, turns)
  - Any CSV with 'moves' and 'winner'/'result' columns
"""

import glob
import io
import os
import sys
import time
from typing import Iterator, Optional, Tuple

import chess
import chess.pgn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False
    print("[WARNING] zstandard not installed — .pgn.zst files will be skipped.")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("[WARNING] pandas not installed — CSV files will be skipped.")

# ==============================================================================
# Configuration & Hyperparameters
# ==============================================================================
CONFIG = {
    # ---- Data Sources (list all files/dirs you want to use) ------------------
    # Supported: .pgn.zst, .pgn, .csv
    # Use glob patterns (e.g., '/kaggle/input/**/*.pgn') or exact paths.
    "data_sources": [
        "/kaggle/input/**/*.pgn.zst",
        "/kaggle/input/**/*.pgn",
        "/kaggle/input/**/*.csv",
    ],

    # ---- Fine-tuning: set to a .pt path to continue from existing weights ----
    # Set to None to train from scratch.
    "pretrained_weights": "/kaggle/input/chesson-weights/chesson_weights.pt",

    # ---- Training hyperparameters -------------------------------------------
    "batch_size": 512,
    "lr": 3e-4,            # Lower LR for fine-tuning (use 1e-3 for scratch)
    "weight_decay": 1e-4,
    "num_res_blocks": 6,
    "channels": 128,
    "max_positions": 3_000_000,  # ~100K games × 30 moves avg — cap for Kaggle quota
    "min_moves_per_game": 15,    # Skip very short/corrupt games
    "elo_filter": 1500,          # Minimum Elo to include (set 0 to disable)

    # ---- Output --------------------------------------------------------------
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "output_weights": "/kaggle/working/chesson_weights.pt",
    "log_interval": 100,         # Print stats every N batches
}

ACTION_SIZE = 4168
NUM_PLANES = 14

PIECE_TO_PLANE = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

# ==============================================================================
# Board & Move Encoding
# ==============================================================================
def encode_board(board: chess.Board) -> np.ndarray:
    """Encode chess.Board into a canonical (14, 8, 8) NumPy tensor.

    Always oriented from the active player's perspective (canonical view).
    Planes 0-5:  active player's pieces (P, N, B, R, Q, K)
    Planes 6-11: opponent's pieces      (p, n, b, r, q, k)
    Plane 12:    castling rights
    Plane 13:    en-passant square
    """
    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)
    canonical = board.mirror() if board.turn == chess.BLACK else board

    for square in chess.SQUARES:
        piece = canonical.piece_at(square)
        if piece is not None:
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            plane_idx = PIECE_TO_PLANE[piece.piece_type]
            if piece.color == chess.BLACK:
                plane_idx += 6
            planes[plane_idx, rank, file] = 1.0

    # Castling rights (Plane 12)
    if canonical.has_kingside_castling_rights(chess.WHITE):
        planes[12, 0, 4] = planes[12, 0, 7] = 1.0
    if canonical.has_queenside_castling_rights(chess.WHITE):
        planes[12, 0, 4] = planes[12, 0, 0] = 1.0
    if canonical.has_kingside_castling_rights(chess.BLACK):
        planes[12, 7, 4] = planes[12, 7, 7] = 1.0
    if canonical.has_queenside_castling_rights(chess.BLACK):
        planes[12, 7, 4] = planes[12, 7, 0] = 1.0

    # En passant (Plane 13)
    if canonical.ep_square is not None:
        planes[13, chess.square_rank(canonical.ep_square), chess.square_file(canonical.ep_square)] = 1.0

    return planes


def move_to_index(move: chess.Move, turn: chess.Color) -> int:
    """Convert a chess.Move to an action index in [0, ACTION_SIZE)."""
    if turn == chess.BLACK:
        from_sq = chess.square_mirror(move.from_square)
        to_sq = chess.square_mirror(move.to_square)
    else:
        from_sq = move.from_square
        to_sq = move.to_square

    if move.promotion is not None and move.promotion != chess.QUEEN:
        from_file = chess.square_file(from_sq)
        to_file = chess.square_file(to_sq)
        slot = (from_sq - 48) * 3 + (to_file - from_file + 1)
        if move.promotion == chess.KNIGHT:
            return 4096 + slot
        elif move.promotion == chess.BISHOP:
            return 4096 + 24 + slot
        elif move.promotion == chess.ROOK:
            return 4096 + 48 + slot

    return from_sq * 64 + to_sq


# ==============================================================================
# Game Generators (one per format)
# ==============================================================================
def _result_to_value(result: str, is_white_turn: bool) -> Optional[float]:
    """Return the value target for the current player, or None to skip."""
    if result == "1-0":
        return 1.0 if is_white_turn else -1.0
    elif result == "0-1":
        return -1.0 if is_white_turn else 1.0
    elif result == "1/2-1/2":
        return 0.0
    return None  # Unknown result — skip


def _yield_from_game(
    game: chess.pgn.Game,
    result: str,
    min_moves: int,
    elo_filter: int,
) -> Iterator[Tuple[np.ndarray, int, float]]:
    """Yield (board_tensor, move_idx, value) triples from a single PGN game."""
    moves = list(game.mainline_moves())
    if len(moves) < min_moves:
        return

    # Optional Elo filter
    if elo_filter > 0:
        try:
            white_elo = int(game.headers.get("WhiteElo", "0") or "0")
            black_elo = int(game.headers.get("BlackElo", "0") or "0")
            if white_elo < elo_filter or black_elo < elo_filter:
                return
        except (ValueError, TypeError):
            pass  # No Elo headers — allow game through

    board = game.board()
    for move in moves:
        if not board.is_legal(move):
            break
        is_white_turn = board.turn == chess.WHITE
        val = _result_to_value(result, is_white_turn)
        if val is not None:
            yield encode_board(board), move_to_index(move, board.turn), np.float32(val)
        board.push(move)


def stream_pgn_zst(path: str, min_moves: int, elo_filter: int) -> Iterator[Tuple]:
    """Stream positions from a .pgn.zst compressed file."""
    if not HAS_ZSTD:
        print(f"[SKIP] zstandard not available — cannot read {path}")
        return
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
            while True:
                try:
                    game = chess.pgn.read_game(text_stream)
                except Exception:
                    continue
                if game is None:
                    break
                result = game.headers.get("Result", "*")
                if result not in ("1-0", "0-1", "1/2-1/2"):
                    continue
                yield from _yield_from_game(game, result, min_moves, elo_filter)


def stream_pgn(path: str, min_moves: int, elo_filter: int) -> Iterator[Tuple]:
    """Stream positions from a plain .pgn file."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        while True:
            try:
                game = chess.pgn.read_game(f)
            except Exception:
                continue
            if game is None:
                break
            result = game.headers.get("Result", "*")
            if result not in ("1-0", "0-1", "1/2-1/2"):
                continue
            yield from _yield_from_game(game, result, min_moves, elo_filter)


def stream_csv(path: str, min_moves: int, elo_filter: int) -> Iterator[Tuple]:
    """Stream positions from a Kaggle CSV chess games file.

    Supports the popular 'games.csv' format from datasnaek/chess dataset:
      - 'moves': space-separated UCI move string (e.g. "e2e4 e7e5 ...")
      - 'winner': 'white', 'black', or 'draw'
      - 'turns': number of half-moves
      - Optional 'white_rating', 'black_rating' for Elo filtering

    Also tries to auto-detect 'result' column for +1/-1/draw style results.
    """
    if not HAS_PANDAS:
        print(f"[SKIP] pandas not available — cannot read {path}")
        return

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[SKIP] Could not read CSV {path}: {e}")
        return

    # Detect move column
    move_col = None
    for col in ["moves", "pgn", "game", "uci"]:
        if col in df.columns:
            move_col = col
            break
    if move_col is None:
        print(f"[SKIP] No recognized moves column in {path}. Columns: {list(df.columns)}")
        return

    # Detect result column — try 'winner' first, then 'result'
    def get_result(row) -> Optional[str]:
        if "winner" in df.columns:
            w = str(row.get("winner", "")).strip().lower()
            return {"white": "1-0", "black": "0-1", "draw": "1/2-1/2"}.get(w)
        if "result" in df.columns:
            r = str(row.get("result", "")).strip()
            if r in ("1-0", "0-1", "1/2-1/2"):
                return r
        return None

    for _, row in df.iterrows():
        result = get_result(row)
        if result is None:
            continue

        moves_str = str(row[move_col]).strip()
        if not moves_str:
            continue

        # Parse moves — try UCI first, then SAN
        board = chess.Board()
        parsed_moves = []
        try:
            for tok in moves_str.split():
                try:
                    move = chess.Move.from_uci(tok)
                    if board.is_legal(move):
                        parsed_moves.append(move)
                        board.push(move)
                    else:
                        break
                except ValueError:
                    # Try SAN
                    try:
                        move = board.parse_san(tok)
                        parsed_moves.append(move)
                        board.push(move)
                    except Exception:
                        break
        except Exception:
            continue

        if len(parsed_moves) < min_moves:
            continue

        # Optional Elo filter from CSV rating columns
        if elo_filter > 0:
            wr = int(row.get("white_rating", row.get("WhiteElo", 0)) or 0)
            br = int(row.get("black_rating", row.get("BlackElo", 0)) or 0)
            if (wr > 0 and wr < elo_filter) or (br > 0 and br < elo_filter):
                continue

        board = chess.Board()
        for move in parsed_moves:
            is_white_turn = board.turn == chess.WHITE
            val = _result_to_value(result, is_white_turn)
            if val is not None:
                yield encode_board(board), move_to_index(move, board.turn), np.float32(val)
            board.push(move)


# ==============================================================================
# Multi-Source Streaming Dataset
# ==============================================================================
class MultiSourceChessDataset(IterableDataset):
    """Streams chess positions from multiple files in multiple formats.

    Automatically detects file format (.pgn.zst, .pgn, .csv) and streams
    all positions up to ``max_positions`` total.
    """

    def __init__(
        self,
        data_sources: list,
        max_positions: int = 3_000_000,
        min_moves: int = 15,
        elo_filter: int = 0,
    ):
        self.max_positions = max_positions
        self.min_moves = min_moves
        self.elo_filter = elo_filter

        # Expand glob patterns into concrete file paths
        self.files = []
        for src in data_sources:
            if "*" in src or "?" in src:
                matched = sorted(glob.glob(src, recursive=True))
                self.files.extend(matched)
            elif os.path.isfile(src):
                self.files.append(src)
            elif os.path.isdir(src):
                for ext in (".pgn.zst", ".pgn", ".csv"):
                    self.files.extend(glob.glob(os.path.join(src, f"**/*{ext}"), recursive=True))

        # Deduplicate while preserving order
        seen = set()
        self.files = [f for f in self.files if not (f in seen or seen.add(f))]

        if not self.files:
            print("[WARNING] No data files found! Check your data_sources paths.")
        else:
            print(f"[Dataset] Found {len(self.files)} file(s):")
            for f in self.files:
                print(f"  - {os.path.basename(f)}")

    def _stream_file(self, path: str) -> Iterator[Tuple]:
        """Route to the correct parser based on file extension."""
        if path.endswith(".pgn.zst"):
            yield from stream_pgn_zst(path, self.min_moves, self.elo_filter)
        elif path.endswith(".pgn"):
            yield from stream_pgn(path, self.min_moves, self.elo_filter)
        elif path.endswith(".csv"):
            yield from stream_csv(path, self.min_moves, self.elo_filter)
        else:
            print(f"[SKIP] Unknown file format: {path}")

    def __iter__(self):
        count = 0
        for path in self.files:
            print(f"\n[Dataset] Streaming: {os.path.basename(path)}")
            for sample in self._stream_file(path):
                yield sample
                count += 1
                if count >= self.max_positions:
                    print(f"\n[Dataset] Reached max_positions={self.max_positions:,}. Stopping.")
                    return
        print(f"\n[Dataset] Exhausted all files. Total positions: {count:,}")


# ==============================================================================
# Model Architecture (matches chesson/model.py exactly)
# ==============================================================================
class ResidualBlock(nn.Module):
    """Residual block with GroupNorm and SiLU activation."""

    def __init__(self, channels: int = 128):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class ChesSonNet(nn.Module):
    """Dual-head ResNet for chess move policy and position value."""

    def __init__(self, num_blocks: int = 6, channels: int = 128):
        super().__init__()
        self.input = nn.Sequential(
            nn.Conv2d(NUM_PLANES, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, channels),
            nn.SiLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])
        self.policy = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=1, bias=False),
            nn.GroupNorm(4, 64),
            nn.SiLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, ACTION_SIZE),
        )
        self.value = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1, bias=False),
            nn.GroupNorm(4, 32),
            nn.SiLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        x = self.input(x)
        x = self.res_blocks(x)
        return self.policy(x), self.value(x)


# ==============================================================================
# Training Loop
# ==============================================================================
def train():
    device = torch.device(CONFIG["device"])
    print(f"Using device: {device}")

    # Initialize model
    model = ChesSonNet(num_blocks=CONFIG["num_res_blocks"], channels=CONFIG["channels"]).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ChesSonNet — {total_params:,} trainable parameters")

    # Fine-tuning: load pretrained weights if specified
    pretrained = CONFIG.get("pretrained_weights")
    if pretrained and os.path.isfile(pretrained):
        try:
            state = torch.load(pretrained, map_location=device)
            model.load_state_dict(state, strict=True)
            print(f"[Fine-tune] Loaded pretrained weights from: {pretrained}")
        except Exception as e:
            print(f"[WARNING] Could not load weights from {pretrained}: {e}")
            print("[WARNING] Falling back to training from scratch.")
    else:
        if pretrained:
            print(f"[INFO] Pretrained weights not found at {pretrained} — training from scratch.")
        else:
            print("[INFO] No pretrained weights specified — training from scratch.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    # Build dataset
    dataset = MultiSourceChessDataset(
        data_sources=CONFIG["data_sources"],
        max_positions=CONFIG["max_positions"],
        min_moves=CONFIG["min_moves_per_game"],
        elo_filter=CONFIG.get("elo_filter", 0),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    print(f"\nStarting training... (max {CONFIG['max_positions']:,} positions)")
    model.train()

    total_loss = 0.0
    correct_top1, correct_top3 = 0, 0
    total_samples = 0
    start_time = time.time()

    log_interval = CONFIG.get("log_interval", 100)
    pbar = tqdm(dataloader, desc="chesSon Training")

    for batch_idx, (boards, move_targets, val_targets) in enumerate(pbar):
        boards = boards.to(device, non_blocking=True)
        move_targets = move_targets.to(device, non_blocking=True)
        val_targets = val_targets.unsqueeze(1).to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            p_logits, v_pred = model(boards)
            loss_policy = policy_criterion(p_logits, move_targets)
            loss_value = value_criterion(v_pred, val_targets)
            loss = loss_policy + 0.5 * loss_value

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            _, top3_preds = p_logits.topk(3, dim=1)
            correct_top1 += (top3_preds[:, 0] == move_targets).sum().item()
            correct_top3 += (top3_preds == move_targets.unsqueeze(1)).any(dim=1).sum().item()
            total_samples += move_targets.size(0)
            total_loss += loss.item()

        if (batch_idx + 1) % log_interval == 0:
            top1_acc = 100.0 * correct_top1 / total_samples
            top3_acc = 100.0 * correct_top3 / total_samples
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({
                "Loss": f"{avg_loss:.3f}",
                "Pol": f"{loss_policy.item():.3f}",
                "Val": f"{loss_value.item():.3f}",
                "Top1%": f"{top1_acc:.1f}",
                "Top3%": f"{top3_acc:.1f}",
            })

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed / 60:.2f} minutes.")
    if total_samples > 0:
        print(f"Final Top-1 Accuracy: {100.0 * correct_top1 / total_samples:.2f}%")
        print(f"Final Top-3 Accuracy: {100.0 * correct_top3 / total_samples:.2f}%")
        print(f"Total positions trained: {total_samples:,}")

    torch.save(model.state_dict(), CONFIG["output_weights"])
    print(f"\nWeights saved to: {CONFIG['output_weights']}")
    print("Download from Kaggle Output tab → place into chesSon/weights/chesson_weights.pt")


if __name__ == "__main__":
    train()
