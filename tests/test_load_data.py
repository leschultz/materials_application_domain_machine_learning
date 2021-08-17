from mad.datasets import load_data
import unittest


class ml_test(unittest.TestCase):

    def test_loads(self):
        '''
        Test data import
        '''

        load_data.friedman()
        load_data.diffusion()
        load_data.super_cond()


if __name__ == '__main__':
    unittest.main()
