"""Enhanced chesSon Lichess Bot connector.

Features:
- Beautiful colored terminal interface with ASCII board, evaluation bar, and piece movements.
- Configurable game time (Blitz, Rapid, Classical, Correspondence) with increment.
- Clear move logs specifying exact piece ("White Knight", "Black Queen") and captures.
- Choice of bot piece color (White, Black, or Random).
- Direct challenge to Lichess Stockfish levels 1-8 or open lobby for human challenges.
"""

import os
import sys
import time
import argparse
import chess
import berserk
from colorama import init, Fore, Back, Style

init(autoreset=True)

from chesson.engine import load_model
from chesson.search import ChesSonSearcher

PIECE_NAMES = {
    chess.PAWN: "Pawn",
    chess.KNIGHT: "Knight",
    chess.BISHOP: "Bishop",
    chess.ROOK: "Rook",
    chess.QUEEN: "Queen",
    chess.KING: "King",
}

PIECE_SYMBOLS = {
    "P": f"{Fore.WHITE}♙{Style.RESET_ALL}",
    "N": f"{Fore.WHITE}♘{Style.RESET_ALL}",
    "B": f"{Fore.WHITE}♗{Style.RESET_ALL}",
    "R": f"{Fore.WHITE}♖{Style.RESET_ALL}",
    "Q": f"{Fore.WHITE}♕{Style.RESET_ALL}",
    "K": f"{Fore.WHITE}♔{Style.RESET_ALL}",
    "p": f"{Fore.LIGHTCYAN_EX}♟{Style.RESET_ALL}",
    "n": f"{Fore.LIGHTCYAN_EX}♞{Style.RESET_ALL}",
    "b": f"{Fore.LIGHTCYAN_EX}♝{Style.RESET_ALL}",
    "r": f"{Fore.LIGHTCYAN_EX}♜{Style.RESET_ALL}",
    "q": f"{Fore.LIGHTCYAN_EX}♛{Style.RESET_ALL}",
    "k": f"{Fore.LIGHTCYAN_EX}♚{Style.RESET_ALL}",
    ".": f"{Fore.BLACK}·{Style.RESET_ALL}",
}


def describe_move(board: chess.Board, move: chess.Move) -> str:
    """Returns a rich human-readable description of the move (piece name, from-to, captures, checks)."""
    piece = board.piece_at(move.from_square)
    if piece is None:
        return move.uci()

    color_str = f"{Fore.WHITE}White{Style.RESET_ALL}" if piece.color == chess.WHITE else f"{Fore.LIGHTCYAN_EX}Black{Style.RESET_ALL}"
    piece_name = PIECE_NAMES.get(piece.piece_type, "Piece")
    from_sq = chess.square_name(move.from_square)
    to_sq = chess.square_name(move.to_square)

    is_capture = board.is_capture(move)
    captured_piece = board.piece_at(move.to_square) if is_capture else None

    # Castling
    if board.is_castling(move):
        if chess.square_file(move.to_square) > chess.square_file(move.from_square):
            action = f"{Fore.YELLOW}Kingside Castles (O-O){Style.RESET_ALL}"
        else:
            action = f"{Fore.YELLOW}Queenside Castles (O-O-O){Style.RESET_ALL}"
        return f"{color_str} {piece_name} {action}"

    if is_capture:
        cap_name = PIECE_NAMES.get(captured_piece.piece_type, "piece") if captured_piece else "Pawn"
        action = f"moves from {Fore.LIGHTBLUE_EX}{from_sq}{Style.RESET_ALL} and {Fore.RED}CAPTURES{Style.RESET_ALL} {cap_name} on {Fore.LIGHTBLUE_EX}{to_sq}{Style.RESET_ALL}"
    else:
        action = f"moves from {Fore.LIGHTBLUE_EX}{from_sq}{Style.RESET_ALL} ➔ {Fore.LIGHTBLUE_EX}{to_sq}{Style.RESET_ALL}"

    desc = f"{color_str} {Fore.GREEN}{piece_name}{Style.RESET_ALL} {action}"

    if move.promotion:
        promo_name = PIECE_NAMES.get(move.promotion, "Queen")
        desc += f" (Promoted to {Fore.MAGENTA}{promo_name}{Style.RESET_ALL}!)"

    # Check state after move
    board.push(move)
    if board.is_checkmate():
        desc += f" {Fore.RED}{Style.BRIGHT}# CHECKMATE!{Style.RESET_ALL}"
    elif board.is_check():
        desc += f" {Fore.YELLOW}+ CHECK!{Style.RESET_ALL}"
    board.pop()

    return desc


def render_board(board: chess.Board, my_color: chess.Color, last_move: chess.Move = None) -> str:
    """Renders a stylish 8x8 chess board with coordinates and colors."""
    lines = []
    lines.append(f"    {Fore.LIGHTBLACK_EX}a   b   c   d   e   f   g   h{Style.RESET_ALL}")
    lines.append(f"  {Fore.LIGHTBLACK_EX}┌───┬───┬───┬───┬───┬───┬───┬───┐{Style.RESET_ALL}")

    ranks = range(7, -1, -1) if my_color == chess.WHITE else range(0, 8)
    files = range(0, 8) if my_color == chess.WHITE else range(7, -1, -1)

    for r in ranks:
        row_str = f"{Fore.LIGHTBLACK_EX}{r + 1} │{Style.RESET_ALL}"
        for f in files:
            sq = chess.square(f, r)
            p = board.piece_at(sq)
            sym = PIECE_SYMBOLS.get(p.symbol(), ".") if p else f"{Fore.LIGHTBLACK_EX}·{Style.RESET_ALL}"

            # Highlight last moved squares
            if last_move and (sq == last_move.from_square or sq == last_move.to_square):
                row_str += f" {Back.MAGENTA}{sym}{Style.RESET_ALL}{Back.RESET} {Fore.LIGHTBLACK_EX}│{Style.RESET_ALL}"
            else:
                row_str += f" {sym} {Fore.LIGHTBLACK_EX}│{Style.RESET_ALL}"

        lines.append(row_str + f" {Fore.LIGHTBLACK_EX}{r + 1}{Style.RESET_ALL}")
        if r != (0 if my_color == chess.WHITE else 7):
            lines.append(f"  {Fore.LIGHTBLACK_EX}├───┼───┼───┼───┼───┼───┼───┼───┤{Style.RESET_ALL}")

    lines.append(f"  {Fore.LIGHTBLACK_EX}└───┴───┴───┴───┴───┴───┴───┴───┘{Style.RESET_ALL}")
    lines.append(f"    {Fore.LIGHTBLACK_EX}a   b   c   d   e   f   g   h{Style.RESET_ALL}")
    return "\n".join(lines)


def run_lichess_bot(token: str, weights_path: str, challenge_ai_level: int = None, color: str = "white", clock_limit: int = 180, clock_increment: int = 2):
    session = berserk.TokenSession(token)
    client = berserk.Client(session=session)

    print(f"\n{Fore.CYAN}============================================================={Style.RESET_ALL}")
    print(f"{Fore.GREEN}{Style.BRIGHT}                ♞ chesSon Neural Chess Bot ♞                 {Style.RESET_ALL}")
    print(f"{Fore.CYAN}============================================================={Style.RESET_ALL}")

    try:
        user_info = client.account.get()
        bot_tag = f"{Fore.GREEN}[BOT]{Style.RESET_ALL}" if user_info.get("title") == "BOT" else f"{Fore.YELLOW}[USER]{Style.RESET_ALL}"
        print(f"Logged in as: {Fore.YELLOW}{Style.BRIGHT}@{user_info.get('username')}{Style.RESET_ALL} {bot_tag}")
    except Exception as e:
        print(f"{Fore.RED}Authentication failed: {e}{Style.RESET_ALL}")
        return

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading chesSon weights from: {Fore.BLUE}{weights_path}{Style.RESET_ALL}")
    model = load_model(weights_path=weights_path, device=device)
    searcher = ChesSonSearcher(model=model, device=device, max_time=3.0)
    print(f"Inference device: {Fore.GREEN}{device.upper()}{Style.RESET_ALL}")

    if challenge_ai_level is not None:
        print(f"\n{Fore.MAGENTA}Challenging Stockfish AI Level {challenge_ai_level}...{Style.RESET_ALL}")
        print(f"Time Control: {Fore.CYAN}{clock_limit // 60} min + {clock_increment}s increment{Style.RESET_ALL} | Color: {Fore.YELLOW}{color.upper()}{Style.RESET_ALL}")
        try:
            # Challenge AI with clock control
            challenge_res = client.challenges.create_ai(
                level=challenge_ai_level,
                clock_limit=clock_limit,
                clock_increment=clock_increment,
                color=color,
                variant="standard"
            )
            game_id = challenge_res.get("id")
            print(f"{Fore.GREEN}Challenge sent successfully! Game ID: {game_id}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}═════════════════════════════════════════════════════════════{Style.RESET_ALL}")
            print(f"{Fore.GREEN}{Style.BRIGHT}  GAME STARTED! Live view: https://lichess.org/{game_id}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}═════════════════════════════════════════════════════════════{Style.RESET_ALL}")
            play_game(client, searcher, game_id, user_info["id"])
            return
        except Exception as err:
            print(f"{Fore.RED}Could not challenge AI: {err}{Style.RESET_ALL}")

    print(f"\n{Fore.GREEN}Online & waiting for games...{Style.RESET_ALL} (Press Ctrl+C to stop)")
    print(f"{Fore.LIGHTBLACK_EX}─────────────────────────────────────────────────────────────{Style.RESET_ALL}")

    for event in client.bots.stream_incoming_events():
        event_type = event.get("type")

        if event_type == "challenge":
            challenge = event["challenge"]
            c_id = challenge["id"]
            challenger = challenge.get("challenger", {}).get("name", "Unknown")
            print(f"\n{Fore.YELLOW}Challenge received from @{challenger} (ID: {c_id}) ➜ Auto-accepting...{Style.RESET_ALL}")
            client.bots.accept_challenge(c_id)

        elif event_type == "gameStart":
            game_id = event["game"]["id"]
            print(f"\n{Fore.CYAN}═════════════════════════════════════════════════════════════{Style.RESET_ALL}")
            print(f"{Fore.GREEN}{Style.BRIGHT}  GAME STARTED! Live view: https://lichess.org/{game_id}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}═════════════════════════════════════════════════════════════{Style.RESET_ALL}")
            play_game(client, searcher, game_id, user_info["id"])


def play_game(client, searcher, game_id, bot_id):
    board = chess.Board()
    my_color = None
    last_move = None
    move_count = 0
    clock_limit = 180   # default 3 min
    clock_increment = 2
    is_bullet = False
    BULLET_POLICY_MOVES = 12  # use pure policy for first N half-moves in bullet

    for event in client.bots.stream_game_state(game_id):
        e_type = event.get("type")

        if e_type == "gameFull":
            white_id = event.get("white", {}).get("id", "").lower()
            white_name = event.get("white", {}).get("name", "White")
            black_name = event.get("black", {}).get("name", "Black")
            my_color = chess.WHITE if white_id == bot_id.lower() else chess.BLACK

            # Extract clock from gameFull
            clock_info = event.get("clock", {})
            clock_limit = clock_info.get("initial", 180000) // 1000   # ms → s
            clock_increment = clock_info.get("increment", 2000) // 1000
            is_bullet = clock_limit <= 60

            tc_label = "⚡ BULLET" if is_bullet else ("🔥 BLITZ" if clock_limit <= 300 else "🕐 RAPID")
            print(f"Time Control: {Fore.CYAN}{tc_label} ({clock_limit // 60}m+{clock_increment}s){Style.RESET_ALL}")
            print(f"Match: {Fore.WHITE}{white_name} (White){Style.RESET_ALL} vs {Fore.LIGHTCYAN_EX}{black_name} (Black){Style.RESET_ALL}")
            print(f"chesSon is playing as: {Fore.YELLOW}{Style.BRIGHT}{'WHITE' if my_color == chess.WHITE else 'BLACK'}{Style.RESET_ALL}")
            if is_bullet:
                print(f"{Fore.MAGENTA}⚡ Bullet mode: pure-policy for first {BULLET_POLICY_MOVES} half-moves{Style.RESET_ALL}")
            print()

            moves_str = event.get("state", {}).get("moves", "")
        elif e_type == "gameState":
            moves_str = event.get("moves", "")
            status = event.get("status")
            winner = event.get("winner")
            if status not in ("started", "created"):
                print(f"\n{Fore.CYAN}─────────────────────────────────────────────────────────────{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}{Style.BRIGHT}Game Over! Status: {status.upper()} | Winner: {winner.upper() if winner else 'DRAW'}{Style.RESET_ALL}")
                print(f"{Fore.CYAN}─────────────────────────────────────────────────────────────{Style.RESET_ALL}\n")
                break
        elif e_type == "chatLine":
            continue
        else:
            continue

        # Replay moves
        board.reset()
        moves_list = moves_str.split() if moves_str else []

        if len(moves_list) > move_count:
            # New move happened
            new_moves = moves_list[move_count:]
            temp_board = chess.Board()
            for m in moves_list[:move_count]:
                temp_board.push_uci(m)

            for uci_str in new_moves:
                mv = chess.Move.from_uci(uci_str)
                mv_desc = describe_move(temp_board, mv)
                turn_no = temp_board.fullmove_number
                turn_label = f"#{turn_no}" if temp_board.turn == chess.WHITE else f"#{turn_no}..."
                
                # Print move banner
                if temp_board.turn == my_color:
                    print(f"\n{Fore.GREEN}🤖 chesSon {turn_label}: {mv_desc}{Style.RESET_ALL}")
                else:
                    print(f"\n{Fore.MAGENTA}⚔️ Opponent {turn_label}: {mv_desc}{Style.RESET_ALL}")

                temp_board.push(mv)
                last_move = mv

            move_count = len(moves_list)

        # Sync board
        for uci_str in moves_list:
            try:
                board.push_uci(uci_str)
            except Exception:
                pass

        # Print current board view
        if last_move:
            print(render_board(board, my_color, last_move))
            _, score = searcher.evaluate_position(board)
            score_bar = f"{Fore.GREEN}+{score:.2f}{Style.RESET_ALL}" if score >= 0 else f"{Fore.RED}{score:.2f}{Style.RESET_ALL}"
            print(f"Board Eval (chesSon Net): [{score_bar}]\n")

        # Extract remaining clock time from gameState
        if e_type in ("gameFull", "gameState"):
            state_data = event.get("state", event) if e_type == "gameFull" else event
            if my_color == chess.WHITE:
                my_time_left = state_data.get("wtime", clock_limit * 1000) / 1000.0
            else:
                my_time_left = state_data.get("btime", clock_limit * 1000) / 1000.0
        else:
            my_time_left = clock_limit  # fallback

        # It's our turn to move!
        if board.turn == my_color and not board.is_game_over():
            # Set adaptive time budget
            searcher.set_time_budget(
                seconds_left=my_time_left,
                increment=clock_increment,
                move_number=board.fullmove_number,
            )
            budget_display = f"{searcher.max_time:.2f}s"
            clock_display = f"{my_time_left:.1f}s left"
            print(f"{Fore.YELLOW}⏳ chesSon thinking (budget: {budget_display}, clock: {clock_display})...{Style.RESET_ALL}")
            start_t = time.time()
            best_move = searcher.search_best_move(board, depth=2)
            think_time = time.time() - start_t

            # Add a slight natural pacing delay for early moves so it doesn't blitz out robotic instant moves
            if board.fullmove_number <= 8 and my_time_left > 15:
                min_wait = 1.0 if not is_bullet else 0.5
                if think_time < min_wait:
                    time.sleep(min_wait - think_time)
                    think_time = min_wait

            if best_move and best_move in board.legal_moves:
                move_uci = best_move.uci()
                print(f"{Fore.CYAN}Selected move in {think_time:.2f}s ➔ {move_uci}{Style.RESET_ALL}")
                client.bots.make_move(game_id, move_uci)
            else:
                legal = list(board.legal_moves)
                if legal:
                    fallback = legal[0].uci()
                    print(f"{Fore.RED}Fallback move ➔ {fallback}{Style.RESET_ALL}")
                    client.bots.make_move(game_id, fallback)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced chesSon Lichess Bot")
    parser.add_argument("--token", type=str, required=True, help="Lichess Bot Personal API Access Token")
    parser.add_argument("--weights", type=str, default="chesson/weights/chesson_weights.pt", help="Path to weights.pt")
    parser.add_argument("--level", type=int, default=None, choices=range(1, 9), help="Challenge Stockfish level (1-8)")
    parser.add_argument("--color", type=str, default="white", choices=["white", "black", "random"], help="Color against AI")
    parser.add_argument("--time", type=int, default=180, help="Initial game time in seconds (e.g. 180 = 3 min, 300 = 5 min, 600 = 10 min)")
    parser.add_argument("--increment", type=int, default=2, help="Time increment per move in seconds (e.g. 0, 2, 5)")
    args = parser.parse_args()

    run_lichess_bot(
        token=args.token,
        weights_path=args.weights,
        challenge_ai_level=args.level,
        color=args.color,
        clock_limit=args.time,
        clock_increment=args.increment,
    )
