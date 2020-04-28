import unittest

from convert_network import to_fixed_point


NUM_DIGITS = 7


class FixedPointTests(unittest.TestCase):

    def test_powers_of_two_positive(self):
        self.assertEqual(64, to_fixed_point(0.5, NUM_DIGITS))
        self.assertEqual(32, to_fixed_point(0.25, NUM_DIGITS))
        self.assertEqual(16, to_fixed_point(0.125, NUM_DIGITS))
        self.assertEqual(8, to_fixed_point(0.0625, NUM_DIGITS))
        self.assertEqual(4, to_fixed_point(0.03125, NUM_DIGITS))
        self.assertEqual(2, to_fixed_point(0.015625, NUM_DIGITS))
        self.assertEqual(1, to_fixed_point(0.0078125, NUM_DIGITS))
        self.assertEqual(0, to_fixed_point(0.00390625, NUM_DIGITS))

    def test_powers_of_two_negative(self):
        self.assertEqual(-64, to_fixed_point(-0.5, NUM_DIGITS))
        self.assertEqual(-32, to_fixed_point(-0.25, NUM_DIGITS))
        self.assertEqual(-16, to_fixed_point(-0.125, NUM_DIGITS))
        self.assertEqual(-8, to_fixed_point(-0.0625, NUM_DIGITS))
        self.assertEqual(-4, to_fixed_point(-0.03125, NUM_DIGITS))
        self.assertEqual(-2, to_fixed_point(-0.015625, NUM_DIGITS))
        self.assertEqual(-1, to_fixed_point(-0.0078125, NUM_DIGITS))
        self.assertEqual(0, to_fixed_point(-0.00390625, NUM_DIGITS))

    def test_others(self):
        self.assertEqual(41, to_fixed_point(0.32501, NUM_DIGITS))
        self.assertEqual(3, to_fixed_point(0.0267, NUM_DIGITS))
        self.assertEqual(-41, to_fixed_point(-0.32501, NUM_DIGITS))
        self.assertEqual(-3, to_fixed_point(-0.0267, NUM_DIGITS))
        self.assertEqual(96, to_fixed_point(0.75, NUM_DIGITS))
        self.assertEqual(-96, to_fixed_point(-0.75, NUM_DIGITS))


if __name__ == '__main__':
    unittest.main()

