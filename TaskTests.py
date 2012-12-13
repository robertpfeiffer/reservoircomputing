import unittest
import tasks
from math import *

class TaskTests(unittest.TestCase):

    def test_mso_task(self):
        best_nrsme = tasks.mso_task(1)
        self.assertGreater(pow(10,-4), best_nrsme, 'NRMSE too big: %f' % best_nrsme)
        #pass

    def test_mackey_glass(self):
        best_nrsme = tasks.mackey_glass_task()
        self.assertGreater(0.1, best_nrsme, 'NRMSE too big: %f' % best_nrsme)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_mso_task']
    unittest.main()