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

    def test_narma_task(self):
        best_nrsme = tasks.run_NARMA_task()
        self.assertGreater(0.8, best_nrsme, 'NRMSE too big: %f' % best_nrsme)
        #0.8 ist nicht besonders gut
        
    def test_memory_capacity_task(self):
        mem_capacity = tasks.run_memory_task()
        self.assertGreater(mem_capacity, 5, 'Memory Capacity too small: %f' % mem_capacity)
        #0.8 ist nicht besonders gut
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_mso_task']
    unittest.main()