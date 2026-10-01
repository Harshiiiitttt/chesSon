"""Self-contained training script for chesSon on Kaggle.

This script:
1. Streams compressed .pgn.zst games on-the-fly (zero disk exhaustion).
2. Filters broadcast games (moves > 15, valid outcomes, learns from winner).
3. Encodes positions into 8x8x14 canonical tensors and 4168-action indices.
4. Trains ChesSonNet using mixed precision (FP16) on Kaggle GPU.
5. Saves the final weights as 'chesson_weights.pt'.
"""

import io
import os
import sys
import time
import zstandard as zstd
import chess
import chess.pgn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

# ==============================================================================
# Configuration & Hyperparameters
# ==============================================================================
CONFIG = {
    "pgn_zst_path": "/kaggle/input/lichess-broadcasts/broadcasts.pgn.zst",  # Update to your Kaggle dataset path
    "batch_size": 512,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "num_res_blocks": 6,
    "channels": 128,
    "max_positions": 2_000_000,  # Cap total training positions (adjustable)
    "min_moves_per_game": 15,
    "val_split_ratio": 0.05,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "output_weights": "chesson_weights.pt",
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
    """Encodes chess.Board into (14, 8, 8) canonical perspective float32 array."""
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
    """Converts a chess.Move to an action index in [0, 4167]."""
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
# Streaming Dataset from .pgn.zst
# ==============================================================================
class StreamingChessDataset(IterableDataset):
    """Streams games directly from compressed .pgn.zst without uncompressing to disk."""

    def __init__(self, zst_path: str, max_positions: int = 1_000_000, min_moves: int = 15):
        self.zst_path = zst_path
        self.max_positions = max_positions
        self.min_moves = min_moves

    def __iter__(self):
        if not os.path.exists(self.zst_path):
            raise FileNotFoundError(f"Dataset file not found: {self.zst_path}")

        dctx = zstd.ZstdDecompressor()
        count = 0

        with open(self.zst_path, "rb") as fh:
            with dctx.stream_reader(fh) as reader:
                text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")

                while count < self.max_positions:
                    try:
                        game = chess.pgn.read_game(text_stream)
                    except Exception:
                        continue

                    if game is None:
                        break

                    result = game.headers.get("Result", "*")
                    if result not in ("1-0", "0-1", "1/2-1/2"):
                        continue

                    moves = list(game.mainline_moves())
                    if len(moves) < self.min_moves:
                        continue

                    # Decide training targets based on winner
                    # Winner gets value +1.0, loser gets value -1.0, draw gets 0.0
                    train_white = (result in ("1-0", "1/2-1/2"))
                    train_black = (result in ("0-1", "1/2-1/2"))

                    board = game.board()
                    for move in moves:
                        if not board.is_legal(move):
                            break

                        is_white_turn = (board.turn == chess.WHITE)
                        should_train = (is_white_turn and train_white) or (not is_white_turn and train_black)

                        if should_train:
                            # Value target from current player's perspective
                            if result == "1/2-1/2":
                                val_target = 0.0
                            elif (result == "1-0" and is_white_turn) or (result == "0-1" and not is_white_turn):
                                val_target = 1.0
                            else:
                                val_target = -1.0

                            board_tensor = encode_board(board)
                            move_idx = move_to_index(move, board.turn)

                            yield board_tensor, move_idx, np.float32(val_target)
                            count += 1
                            if count >= self.max_positions:
                                return

                        board.push(move)


# ==============================================================================
# Model Architecture
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
        policy_logits = self.policy(x)
        value = self.value(x)
        return policy_logits, value


# ==============================================================================
# Training Loop
# ==============================================================================
def train():
    device = torch.device(CONFIG["device"])
    print(f"Using device: {device}")

    # Initialize model
    model = ChesSonNet(num_blocks=CONFIG["num_res_blocks"], channels=CONFIG["channels"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    dataset = StreamingChessDataset(
        zst_path=CONFIG["pgn_zst_path"],
        max_positions=CONFIG["max_positions"],
        min_moves=CONFIG["min_moves_per_game"],
    )
    dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], num_workers=2, pin_memory=True)

    print(f"Starting training on streaming data from {CONFIG['pgn_zst_path']}...")
    model.train()

    total_loss = 0.0
    correct_top1 = 0
    correct_top3 = 0
    total_samples = 0
    start_time = time.time()

    pbar = tqdm(dataloader, desc="Training")
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

        # Compute Top-1 and Top-3 accuracy
        with torch.no_grad():
            _, top3_preds = p_logits.topk(3, dim=1)
            correct_top1 += (top3_preds[:, 0] == move_targets).sum().item()
            correct_top3 += (top3_preds == move_targets.unsqueeze(1)).any(dim=1).sum().item()
            total_samples += move_targets.size(0)
            total_loss += loss.item()

        if (batch_idx + 1) % 50 == 0:
            top1_acc = 100.0 * correct_top1 / total_samples
            top3_acc = 100.0 * correct_top3 / total_samples
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({
                "Loss": f"{avg_loss:.3f}",
                "PolLoss": f"{loss_policy.item():.3f}",
                "ValLoss": f"{loss_value.item():.3f}",
                "Top1%": f"{top1_acc:.1f}",
                "Top3%": f"{top3_acc:.1f}",
            })

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed / 60:.2f} minutes.")
    print(f"Final Top-1 Accuracy: {100.0 * correct_top1 / total_samples:.2f}%")
    print(f"Final Top-3 Accuracy: {100.0 * correct_top3 / total_samples:.2f}%")

    torch.save(model.state_dict(), CONFIG["output_weights"])
    print(f"Model saved successfully to: {CONFIG['output_weights']}")


if __name__ == "__main__":
    train()
