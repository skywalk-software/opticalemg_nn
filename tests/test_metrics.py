from unittest import TestCase

import torch

from metrics import process_clicks


class Test(TestCase):
    def test_process_clicks(self):
        y, y_hat = torch.IntTensor([0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0]), \
                   torch.IntTensor([0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0])
        result = process_clicks(y, y_hat)
        print(result)
        total_true_clicks, total_false_clicks, total_missed_clicks, total_detected_clicks, \
            on_set_offsets, off_set_offsets, drops = result
        self.assertEqual(total_true_clicks, 4)
        self.assertEqual(total_false_clicks, 1)
        self.assertEqual(total_missed_clicks, 1)
        self.assertEqual(total_detected_clicks, 3)

        self.assertEqual(on_set_offsets, [-1, 1, -2])
        self.assertEqual(off_set_offsets, [1, 1, -1])
        self.assertEqual(drops, [1, 1])
        print(result)

    def test_random_process_clicks(self):
        y, y_hat = \
            torch.IntTensor([1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0]), \
            torch.IntTensor([1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0])
        result = process_clicks(y, y_hat)
        print(result)
        total_true_clicks, total_false_clicks, total_missed_clicks, total_detected_clicks, \
        on_set_offsets, off_set_offsets, drops = result

        self.assertEqual(total_true_clicks, total_missed_clicks + total_detected_clicks)
        self.assertEqual(len(on_set_offsets), total_detected_clicks)
        self.assertEqual(len(off_set_offsets), total_detected_clicks)