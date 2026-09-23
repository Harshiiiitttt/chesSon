"""Search engine for chesSon.

Key features:
- Zobrist hash table for fast transposition lookup.
- Iterative deepening with a configurable time budget.
- Quiescence search that expands only captures and checks.
- Policy‑guided move ordering plus MVV‑LVA for captures.
- Caches legal move mask per position.
"""

import time
import chess
import numpy as np
import torch
from typing import List, Tuple

from chesson.board_encoding import encode_board_tensor, get_legal_move_mask, move_to_index, ACTION_SIZE

# ---------------------------------------------------------------------------
# Zobrist hashing utilities
# ---------------------------------------------------------------------------

_RANDOM_SEED = 123456789
np.random.seed(_RANDOM_SEED)

# 12 piece types * 64 squares + 4 castling flags + 8 en‑passant files
_ZOBRIST_TABLE = np.random.randint(low=0, high=2**64, size=(12, 64), dtype=np.uint64)
_ZOBRIST_CASTLING = np.random.randint(low=0, high=2**64, size=4, dtype=np.uint64)  # KQkq
_ZOBRIST_EP = np.random.randint(low=0, high=2**64, size=8, dtype=np.uint64)  # file index

def _piece_index(piece: chess.Piece) -> int:
    """Map a chess.Piece to an index 0..11.
    Order: WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK.
    """
    base = 0 if piece.color == chess.WHITE else 6
    return base + piece.piece_type - 1

def compute_zobrist(board: chess.Board) -> np.uint64:
    """Compute a 64‑bit Zobrist hash for the given board.
    The function is pure Python and fast enough for our depth‑2/3 searches.
    """
    h = np.uint64(0)
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            h ^= _ZOBRIST_TABLE[_piece_index(piece), sq]
    # Castling rights – order: White K, White Q, Black k, Black q
    cr = board.castling_rights
    if cr & chess.BB_H1:
        h ^= _ZOBRIST_CASTLING[0]
    if cr & chess.BB_A1:
        h ^= _ZOBRIST_CASTLING[1]
    if cr & chess.BB_H8:
        h ^= _ZOBRIST_CASTLING[2]
    if cr & chess.BB_A8:
        h ^= _ZOBRIST_CASTLING[3]
    # En‑passant file
    if board.ep_square is not None:
        file = chess.square_file(board.ep_square)
        h ^= _ZOBRIST_EP[file]
    return h

# ---------------------------------------------------------------------------
# Search implementation
# ---------------------------------------------------------------------------

class ChesSonSearcher:
    """Negamax search with alpha‑beta, Zobrist transposition table and policy ordering.

    Parameters
    ----------
    model: torch.nn.Module
        The trained `ChesSonNet` model.
    device: str
        ``cpu`` or ``cuda`` – the model is kept on this device.
    max_time: float, optional
        Maximum time (seconds) allowed for a move when using iterative deepening.
    """

    def __init__(self, model: torch.nn.Module, device: str = "cpu", max_time: float = 5.0, top_k: int = 6):
        self.model = model
        self.device = device
        self.max_time = max_time
        self.top_k = top_k  # Only search the best K policy moves (0 = no limit)
        self.model.eval()
        self.transposition: dict[np.uint64, Tuple[int, float]] = {}
        # MVV‑LVA values for capture sorting (piece value table)
        self._piece_value = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0,
        }

    def set_time_budget(self, seconds_left, increment = 0.0, move_number: int = 1):
        """Adaptively set max_time based on clock state.

        Budget formula:
        - Bullet (<=60s):  min(0.3, seconds_left * 0.05) + increment * 0.5
        - Blitz  (<=300s): min(1.0, seconds_left * 0.06) + increment * 0.5
        - Rapid  (>300s):  min(3.0, seconds_left * 0.07) + increment * 0.5
        Early moves get slightly less time to save for complications.
        """
        if hasattr(seconds_left, 'total_seconds'):
            seconds_left = seconds_left.total_seconds()
        else:
            try:
                seconds_left = float(seconds_left)
            except Exception:
                seconds_left = 60.0

        if hasattr(increment, 'total_seconds'):
            increment = increment.total_seconds()
        else:
            try:
                increment = float(increment)
            except Exception:
                increment = 0.0

        if seconds_left <= 5:
            # Emergency: play instantly
            self.max_time = 0.05
            return
        if seconds_left <= 60:
            budget = min(0.3, seconds_left * 0.05)
        elif seconds_left <= 300:
            budget = min(1.0, seconds_left * 0.06)
        else:
            budget = min(3.0, seconds_left * 0.07)
        # Add a fraction of increment
        budget += increment * 0.5
        # Ensure opening moves (first 8 moves) have at least a decent base budget to think deeply
        if move_number <= 8 and seconds_left > 15:
            budget = max(budget, 0.8 if seconds_left > 60 else 0.35)
        self.max_time = max(0.05, budget)

    # ---------------------------------------------------------------------
    # Evaluation helper
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def _nn_evaluate(self, board: chess.Board) -> Tuple[np.ndarray, float]:
        """Run the network on ``board`` and return (policy_logits, value)."""
        dtype = next(self.model.parameters()).dtype
        tensor = encode_board_tensor(board, device=self.device, dtype=dtype)
        tensor = tensor.unsqueeze(0)  # batch dim
        policy, value = self.model(tensor)
        return policy.squeeze(0).cpu().numpy(), float(value.squeeze().cpu().numpy())

    def evaluate_position(self, board: chess.Board) -> Tuple[np.ndarray, float]:
        """Public position evaluation returning (policy_logits, value)."""
        return self._nn_evaluate(board)

    # ---------------------------------------------------------------------
    # Move ordering utilities
    # ---------------------------------------------------------------------
    def _order_moves(self, board: chess.Board, policy_logits: np.ndarray) -> List[chess.Move]:
        """Return a list of legal moves ordered by (policy, MVV‑LVA)."""
        turn = board.turn
        moves = list(board.legal_moves)
        idxs = [move_to_index(m, turn) for m in moves]
        priors = np.array([policy_logits[i] for i in idxs])
        # Capture scores (MVV‑LVA)
        mvv = []
        for m in moves:
            if board.is_capture(m):
                victim = board.piece_at(m.to_square)
                attacker = board.piece_at(m.from_square)
                score = self._piece_value[victim.piece_type] - self._piece_value[attacker.piece_type]
            else:
                score = 0
            mvv.append(score)
        combined = priors + 0.5 * np.array(mvv)
        ordered = [move for _, move in sorted(zip(combined, moves), key=lambda x: x[0], reverse=True)]
        # Top-K pruning: only keep the best K moves to shrink the search tree
        if self.top_k > 0 and len(ordered) > self.top_k:
            ordered = ordered[:self.top_k]
        return ordered

    # ---------------------------------------------------------------------
    # Quiescence search (captures & checks only)
    # ---------------------------------------------------------------------
    def _quiescence(self, board: chess.Board, alpha: float, beta: float) -> float:
        _, stand_pat = self._nn_evaluate(board)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat
        for move in board.legal_moves:
            if not board.is_capture(move) and not board.gives_check(move):
                continue
            board.push(move)
            score = -self._quiescence(board, -beta, -alpha)
            board.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    # ---------------------------------------------------------------------
    # Core negamax with alpha‑beta and transposition table
    # ---------------------------------------------------------------------
    def _negamax(self, board: chess.Board, depth: int, alpha: float, beta: float) -> float:
        if board.is_checkmate():
            return -10.0 - depth
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            return 0.0
        if depth == 0:
            return self._quiescence(board, alpha, beta)
        # Transposition lookup
        h = compute_zobrist(board)
        entry = self.transposition.get(h)
        if entry and entry[0] >= depth:
            return entry[1]
        policy_logits, _ = self._nn_evaluate(board)
        ordered_moves = self._order_moves(board, policy_logits)
        value = -float("inf")
        for move in ordered_moves:
            board.push(move)
            score = -self._negamax(board, depth - 1, -beta, -alpha)
            board.pop()
            if score > value:
                value = score
            if value > alpha:
                alpha = value
            if alpha >= beta:
                break
        self.transposition[h] = (depth, value)
        return value

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def select_move_greedy(self, board: chess.Board) -> chess.Move:
        """Return the highest‑policy legal move (no search)."""
        policy_logits, _ = self._nn_evaluate(board)
        turn = board.turn
        best_move = None
        best_score = -float("inf")
        for move in board.legal_moves:
            idx = move_to_index(move, turn)
            score = policy_logits[idx]
            if score > best_score:
                best_score = score
                best_move = move
        return best_move if best_move else next(board.legal_moves)

    def search_best_move(self, board: chess.Board, depth: int = 2) -> chess.Move:
        """Iterative deepening search respecting ``self.max_time``.
        Returns the best move found within the allotted time.
        """
        start = time.time()
        best_move = None
        best_score = -float("inf")
        for current_depth in range(1, depth + 1):
            if time.time() - start > self.max_time:
                break
            policy_logits, _ = self._nn_evaluate(board)
            ordered_moves = self._order_moves(board, policy_logits)
            for move in ordered_moves:
                board.push(move)
                score = -self._negamax(board, current_depth - 1, -float("inf"), float("inf"))
                board.pop()
                if score > best_score:
                    best_score = score
                    best_move = move
            if best_score > 9.0:
                break
        return best_move if best_move else self.select_move_greedy(board)
