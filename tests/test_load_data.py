from madml import datasets
import unittest


class ml_test(unittest.TestCase):

    def test_loads(self):
        '''
        Test data import for all supported sets.
        '''

        for i in datasets.list_data():
            print(i)
            datasets.load(i)


if __name__ == '__main__':
    unittest.main()
