from mad.datasets import load_data
import unittest


class load_data_test(unittest.TestCase):

    def test_loads(self):
        load_data.friedman()


if __name__ == '__main__':
    unittest.main()

