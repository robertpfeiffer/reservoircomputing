import unittest
import tasks
from math import *
import drone_esn
from drone_esn import DroneESN
from esn_persistence import *
from reservoir import *
from esn_readout import *
import os

class TaskTests(unittest.TestCase):

    def test_mackey_glass(self):
        best_nrsme, best_esn = tasks.mackey_glass_task()
        self.assertGreater(0.1, best_nrsme, 'NRMSE too big: %f' % best_nrsme)
        print
        
    def test_memory_capacity_task(self):
        mem_capacity = tasks.memory_task()
        self.assertGreater(mem_capacity, 4, 'Memory Capacity too small: %f' % mem_capacity)
        print
        
    def test_mso_task(self):
        best_nrsme, best_esn = tasks.mso_task(task_type=1)
        self.assertGreater(pow(10,-4), best_nrsme, 'NRMSE too big: %f' % best_nrsme)
        print
        

    def test_narma_task(self):
        best_nrsme, best_esn = tasks.NARMA_task()
        self.assertGreater(0.8, best_nrsme, 'NRMSE too big: %f' % best_nrsme)
        #0.8 ist nicht besonders gut
        print
        
    def test_drone_esn(self):
        results = drone_esn.example_drone_esn(Plots=False)
        self.assertIsNotNone(results, "drone_esn.example_drone_esn failed")
        
    def test_esn_persistence(self):
        machine = ESN(input_dim=1, output_dim = 100)
        trainer = FeedbackReadout(machine, LinearRegressionReadout(machine))
        save_object(trainer, 'tmp_trainer')              
                      
        trainer2 = load_object('tmp_trainer')
        self.assertEquals(trainer2.machine.ninput, 1)
        
        os.remove('tmp_trainer')
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_mso_task']
    unittest.main()