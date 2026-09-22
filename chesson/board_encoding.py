"""Optimised board and move encoding for chesSon.

Features:
- Vectorised NumPy construction of the 8x8x14 tensor.
- Cached mirror mapping for Black-to-move conversion.
- Torch‑compatible ``encode_board_tensor`` that returns a ``float16`` tensor ready for GPU.
- Pre‑computed promotion lookup tables for fast ``move_to_index`` / ``index_to_move``.
"""

import chess
import numpy as np
import torch
from typing import Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ACTION_SIZE = 4168
NUM_PLANES = 14

# Piece type -> plane index (for the active player). Opponent pieces are offset by 6.
PIECE_TO_PLANE = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

# Pre‑computed mirror map for fast board flipping when Black to move.
_MIRROR_MAP = np.arange(64, dtype=np.int16)
for sq in range(64):
    _MIRROR_MAP[sq] = chess.square_mirror(sq)

# Promotion lookup tables – indexed by (from_sq - 48) * 3 + direction (0:left,1:push,2:right)
# They map directly to the action index range 4096..4167.
_PROMOTION_BASE = 4096
_PROMOTION_OFFSETS = {
    chess.KNIGHT: 0,
    chess.BISHOP: 24,
    chess.ROOK: 48,
}

# ---------------------------------------------------------------------------
# Board encoding
# ---------------------------------------------------------------------------
def _empty_tensor() -> np.ndarray:
    """Return a fresh zero‑filled (14,8,8) float32 array.
    Using a function avoids the mutable‑default‑argument pitfall.
    """
    return np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode ``chess.Board`` into a canonical (14,8,8) NumPy tensor.

    The board is always presented from the perspective of the side to move –
    i.e. the active player is considered "White" and its pieces point "up" the
    board. This canonical view dramatically simplifies the neural net.
    """
    # Mirror if Black to move – vectorised.
    canonical = board.mirror() if board.turn == chess.BLACK else board

    planes = _empty_tensor()

    # Vectorised piece placement using NumPy's advanced indexing.
    # Gather all occupied squares and their piece information.
    squares = np.fromiter(chess.SQUARES, dtype=np.int16)
    pieces = np.vectorize(canonical.piece_at)(squares)
    mask = pieces != None  # noqa: E711
    occupied = squares[mask]
    piece_objs = np.array(pieces[mask], dtype=object)

    # Compute rank/file arrays for each occupied square.
    ranks = np.vectorize(chess.square_rank)(occupied)
    files = np.vectorize(chess.square_file)(occupied)

    # Determine plane index for each piece.
    plane_idxs = np.fromiter(
        [
            PIECE_TO_PLANE[p.piece_type] + (0 if p.color == chess.WHITE else 6)
            for p in piece_objs
        ],
        dtype=np.int8,
    )
    planes[plane_idxs, ranks, files] = 1.0

    # Castling rights – we encode them in a single plane (index 12).
    # A bitmask on the home squares indicates the rights.
    cr_plane = planes[12]
    if canonical.has_kingside_castling_rights(chess.WHITE):
        cr_plane[0, 4] = 1.0  # e1
        cr_plane[0, 7] = 1.0  # h1
    if canonical.has_queenside_castling_rights(chess.WHITE):
        cr_plane[0, 4] = 1.0
        cr_plane[0, 0] = 1.0  # a1
    if canonical.has_kingside_castling_rights(chess.BLACK):
        cr_plane[7, 4] = 1.0  # e8
        cr_plane[7, 7] = 1.0  # h8
    if canonical.has_queenside_castling_rights(chess.BLACK):
        cr_plane[7, 4] = 1.0
        cr_plane[7, 0] = 1.0  # a8

    # En‑passant square – single‑pixel plane.
    if canonical.ep_square is not None:
        ep_rank = chess.square_rank(canonical.ep_square)
        ep_file = chess.square_file(canonical.ep_square)
        planes[13, ep_rank, ep_file] = 1.0

    return planes


def encode_board_tensor(
    board: chess.Board, device: str = "cpu", dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Return a torch tensor ready for the model (defaults to float32)."""
    np_tensor = encode_board(board)
    return torch.from_numpy(np_tensor).to(device=device, dtype=dtype)

# ---------------------------------------------------------------------------
# Move <-> index conversion (4168‑action space)
# ---------------------------------------------------------------------------
def _promotion_index(from_sq: int, to_sq: int, promotion: int) -> int:
    """Calculate the under‑promotion index (4096..4167)."""
    # ``from_sq`` is already in canonical (White‑up) orientation.
    from_file = chess.square_file(from_sq)
    to_file = chess.square_file(to_sq)
    direction = to_file - from_file + 1  # 0:left,1:push,2:right
    slot = (from_sq - 48) * 3 + direction  # 0..71
    offset = _PROMOTION_OFFSETS[promotion]
    return _PROMOTION_BASE + offset + slot


def move_to_index(move: chess.Move, turn: chess.Color) -> int:
    """Convert a ``chess.Move`` to an integer index in ``[0, ACTION_SIZE)``.

    The board is assumed to be in *canonical* orientation (active player is
    White). If ``turn`` is Black we first mirror the move coordinates.
    """
    if turn == chess.BLACK:
        from_sq = _MIRROR_MAP[move.from_square]
        to_sq = _MIRROR_MAP[move.to_square]
    else:
        from_sq = move.from_square
        to_sq = move.to_square

    # Promotion handling – queen promotions fall back to the standard move index.
    if move.promotion is not None and move.promotion != chess.QUEEN:
        return _promotion_index(from_sq, to_sq, move.promotion)

    return from_sq * 64 + to_sq


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """Reverse ``move_to_index``.

    ``board`` is the *current* board (not mirrored). The function mirrors the
    coordinates back if the side to move is Black.
    """
    if index < 4096:
        from_sq = index // 64
        to_sq = index % 64
        promotion = None
        # Detect queen promotion automatically if the pawn reaches the last rank.
        if board.turn == chess.WHITE and chess.square_rank(to_sq) == 7:
            if board.piece_type_at(from_sq) == chess.PAWN:
                promotion = chess.QUEEN
        if board.turn == chess.BLACK and chess.square_rank(to_sq) == 0:
            if board.piece_type_at(from_sq) == chess.PAWN:
                promotion = chess.QUEEN
    else:
        # Under‑promotion bucket.
        rel = index - 4096
        # Determine which piece type it is.
        piece_type = None
        for pt, off in _PROMOTION_OFFSETS.items():
            if rel < off + 24:
                piece_type = pt
                rel -= off
                break
        # ``rel`` now in 0..71 range – decode slot.
        from_offset = rel // 3
        direction = rel % 3 - 1  # -1:left,0:push,1:right
        from_sq = 48 + from_offset
        to_sq = chess.square(from_sq % 8 + direction, 7)
        promotion = piece_type

    # Mirror back if Black to move.
    if board.turn == chess.BLACK:
        from_sq = _MIRROR_MAP[from_sq]
        to_sq = _MIRROR_MAP[to_sq]

    return chess.Move(from_sq, to_sq, promotion=promotion)


def get_legal_move_mask(board: chess.Board) -> np.ndarray:
    """Return a boolean mask (size ``ACTION_SIZE``) where ``True`` marks legal moves.
    """
    mask = np.zeros(ACTION_SIZE, dtype=bool)
    turn = board.turn
    for move in board.legal_moves:
        idx = move_to_index(move, turn)
        mask[idx] = True
    return mask
