"""UCI (Universal Chess Interface) Protocol Handler for chesSon.

Allows chesSon to interface with standard chess GUIs (CuteChess, Arena,
ChessBase) and online bots (such as lichess-bot).
"""

import sys
import chess

from chesson import __author__, __engine_name__, __version__
from chesson.search import ChesSonSearcher


class ChesSonUCI:
    """Implements the standard UCI communication loop."""

    def __init__(self, searcher: ChesSonSearcher, default_depth: int = 2):
        self.searcher = searcher
        self.board = chess.Board()
        self.default_depth = default_depth

    def send_response(self, text: str):
        """Sends line to stdout and flushes."""
        print(text, flush=True)

    def handle_position(self, tokens: list):
        """Parses the 'position' UCI command.
        
        Examples:
            position startpos
            position startpos moves e2e4 e7e5
            position fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 moves e2e4
        """
        if not tokens:
            return

        moves_idx = None
        if "moves" in tokens:
            moves_idx = tokens.index("moves")

        if tokens[0] == "startpos":
            self.board = chess.Board()
        elif tokens[0] == "fen":
            fen_tokens = tokens[1:moves_idx] if moves_idx is not None else tokens[1:]
            self.board = chess.Board(" ".join(fen_tokens))

        if moves_idx is not None:
            for move_str in tokens[moves_idx + 1:]:
                try:
                    move = chess.Move.from_uci(move_str)
                    if move in self.board.legal_moves:
                        self.board.push(move)
                except ValueError:
                    pass

    def handle_go(self, tokens: list):
        """Parses 'go' command and outputs bestmove."""
        depth = self.default_depth
        if "depth" in tokens:
            try:
                d_idx = tokens.index("depth") + 1
                depth = int(tokens[d_idx])
            except (ValueError, IndexError):
                depth = self.default_depth

        best_move = self.searcher.search_best_move(self.board, depth=depth)
        if best_move is None:
            # Fallback if no search move
            legal = list(self.board.legal_moves)
            best_move = legal[0] if legal else chess.Move.null()

        self.send_response(f"bestmove {best_move.uci()}")

    def listen(self):
        """Runs the main stdin reading loop."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            tokens = line.split()
            cmd = tokens[0].lower()
            args = tokens[1:]

            if cmd == "uci":
                self.send_response(f"id name {__engine_name__} v{__version__}")
                self.send_response(f"id author {__author__}")
                self.send_response("uciok")
            elif cmd == "isready":
                self.send_response("readyok")
            elif cmd == "ucinewgame":
                self.board = chess.Board()
                self.searcher.transposition.clear()
            elif cmd == "position":
                self.handle_position(args)
            elif cmd == "go":
                self.handle_go(args)
            elif cmd == "quit":
                break
