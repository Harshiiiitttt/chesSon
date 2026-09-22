import os
import sys
import unittest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import chess
import torch
import numpy as np

from chesson.board_encoding import encode_board, move_to_index, index_to_move, get_legal_move_mask, ACTION_SIZE
from chesson.model import ChesSonNet
from chesson.search import ChesSonSearcher


class TestBoardEncoding(unittest.TestCase):

    def test_encode_shape_and_range(self):
        board = chess.Board()
        encoded = encode_board(board)
        self.assertEqual(encoded.shape, (14, 8, 8))
        self.assertTrue(np.all((encoded >= 0.0) & (encoded <= 1.0)))

    def test_move_roundtrip_white(self):
        board = chess.Board()
        for move in board.legal_moves:
            idx = move_to_index(move, board.turn)
            self.assertTrue(0 <= idx < ACTION_SIZE)
            reconstructed = index_to_move(idx, board)
            self.assertEqual(move, reconstructed)

    def test_move_roundtrip_black(self):
        board = chess.Board()
        board.push_san("e4")  # Now Black to move
        for move in board.legal_moves:
            idx = move_to_index(move, board.turn)
            self.assertTrue(0 <= idx < ACTION_SIZE)
            reconstructed = index_to_move(idx, board)
            self.assertEqual(move, reconstructed)

    def test_legal_mask(self):
        board = chess.Board()
        mask = get_legal_move_mask(board)
        self.assertEqual(mask.shape, (ACTION_SIZE,))
        # Initial position has 20 legal moves
        self.assertEqual(np.sum(mask), 20)


class TestModel(unittest.TestCase):

    def test_forward_pass(self):
        model = ChesSonNet(num_blocks=2, channels=32)
        dummy_input = torch.randn(2, 14, 8, 8)
        policy_logits, value = model(dummy_input)

        self.assertEqual(policy_logits.shape, (2, ACTION_SIZE))
        self.assertEqual(value.shape, (2, 1))
        # Value should be bounded by Tanh in [-1, 1]
        self.assertTrue(torch.all(value >= -1.0) and torch.all(value <= 1.0))


class TestSearch(unittest.TestCase):

    def test_search_selects_legal_move(self):
        model = ChesSonNet(num_blocks=2, channels=32)
        searcher = ChesSonSearcher(model, device="cpu")
        board = chess.Board()

        move = searcher.search_best_move(board, depth=1)
        self.assertIn(move, board.legal_moves)


if __name__ == "__main__":
    unittest.main()
