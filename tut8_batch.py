from netpyne import specs
from netpyne.batch import Batch

def batchTau1Tau2():
        # Create variable of type ordered dictionary (NetPyNE's customized version)
        params = specs.ODict()

        # fill in with parameters to explore and range of values (key has to coincide with a variable in simConfig)
        params['synMechTau1'] = [0.01, 0.1, 1.0]
        params['synMechTau2'] = [1.0, 5.0, 10.0]
        

        # create Batch object with parameters to modify, and specifying files to use
        b = Batch(params=params, cfgFile='tut8_cfg.py', netParamsFile='tut8_netParams.py',)

        # Set output folder, grid method (all param combinations), and run configuration
        b.batchLabel = 'tau1tau2'
        b.saveFolder = 'tut8_data'
        b.method = 'grid'
        b.runCfg = {'type': 'mpi_bulletin',
                            'script': 'tut8_init.py',
                            'skip': True}

        # Run batch simulations
        b.run()

# Main code
if __name__ == '__main__':
        batchTau1Tau2()
