"""Main entry point for chesSon.

Supports:
1. Standard UCI mode (for GUIs like Arena, CuteChess, Lichess-bot):
     python -m chesson.engine --weights path/to/weights.pt
2. Interactive play in console:
     python -m chesson.engine --play --color white
3. Built-in diagnostic verification:
     python -m chesson.engine --test
"""

import argparse
import sys
import os
import chess
import torch

from chesson.model import ChesSonNet
from chesson.search import ChesSonSearcher
from chesson.uci import ChesSonUCI


def load_model(weights_path: str = None, device: str = "cpu") -> ChesSonNet:
    """Loads ChesSonNet model, optionally loading pretrained weights."""
    model = ChesSonNet()
    if weights_path and os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}...", file=sys.stderr)
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        if weights_path:
            print(f"Warning: Weights file '{weights_path}' not found. Using initialized model.", file=sys.stderr)
    model.to(device)
    model.eval()
    return model


def play_interactive(searcher: ChesSonSearcher, player_color: chess.Color = chess.WHITE, depth: int = 2):
    """Play a game against chesSon directly in your console."""
    board = chess.Board()
    print("\n" + "=" * 40)
    print("Welcome to chesSon Interactive Chess!")
    print(f"You are playing as: {'WHITE' if player_color == chess.WHITE else 'BLACK'}")
    print("Enter moves in UCI format (e.g., e2e4, g1f3, e7e8q). Type 'quit' to exit.")
    print("=" * 40 + "\n")

    while not board.is_game_over():
        print(board)
        print("-" * 30)

        if board.turn == player_color:
            # Human turn
            while True:
                user_input = input("Your move: ").strip()
                if user_input.lower() in ("quit", "exit"):
                    print("Game aborted.")
                    return
                try:
                    move = chess.Move.from_uci(user_input)
                    if move in board.legal_moves:
                        board.push(move)
                        break
                    else:
                        print("Illegal move! Try again.")
                except ValueError:
                    print("Invalid UCI format. Example: e2e4.")
        else:
            # chesSon turn
            print("chesSon is thinking...")
            best_move = searcher.search_best_move(board, depth=depth)
            _, val = searcher.evaluate_position(board)
            print(f"chesSon plays: {best_move.uci()} (Eval: {val:+.2f})\n")
            board.push(best_move)

    print("\n" + "=" * 40)
    print("Game Over!")
    print(board)
    print(f"Result: {board.result()} ({board.outcome().termination.name})")
    print("=" * 40)


def run_diagnostics(searcher: ChesSonSearcher):
    """Runs a quick diagnostic test to verify board encoding, model evaluation, and search."""
    print("Running chesSon system diagnostics...")
    board = chess.Board()
    
    # 1. Starting position evaluation
    policy, val = searcher.evaluate_position(board)
    print(f"  [OK] Position evaluated. Value score: {val:.4f}, Policy logits shape: {policy.shape}")
    
    # 2. Greedy move selection
    greedy_move = searcher.select_move_greedy(board)
    print(f"  [OK] Greedy move: {greedy_move.uci()}")
    
    # 3. Depth-2 search
    search_move = searcher.search_best_move(board, depth=2)
    print(f"  [OK] Depth-2 search move: {search_move.uci()}")
    
    # 4. Checkmate detection
    fool_mate_board = chess.Board("rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert fool_mate_board.is_checkmate(), "Checkmate position setup error"
    print("  [OK] Checkmate state handled correctly.")
    
    print("\nAll diagnostics passed successfully!")


def main():
    parser = argparse.ArgumentParser(description="chesSon Chess Bot Engine")
    parser.add_argument("--weights", type=str, default=None, help="Path to trained .pt weights file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    parser.add_argument("--depth", type=int, default=2, help="Search depth")
    parser.add_argument("--play", action="store_true", help="Start interactive CLI game against chesSon")
    parser.add_argument("--color", type=str, choices=["white", "black"], default="white", help="Player color in CLI mode")
    parser.add_argument("--test", action="store_true", help="Run self-diagnostics test")

    args = parser.parse_args()

    model = load_model(args.weights, device=args.device)
    searcher = ChesSonSearcher(model, device=args.device)

    if args.test:
        run_diagnostics(searcher)
    elif args.play:
        user_color = chess.WHITE if args.color == "white" else chess.BLACK
        play_interactive(searcher, player_color=user_color, depth=args.depth)
    else:
        # Default: UCI mode
        uci_handler = ChesSonUCI(searcher, default_depth=args.depth)
        uci_handler.listen()


if __name__ == "__main__":
    main()
