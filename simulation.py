#import os
import pexpect
#import subprocess as sp
import gc
import numpy as np
from utils import safe_remove


def compute_coeff(airfoil, reynolds=500000, mach=0, alpha=3, n_iter=200):
    
    gc.collect()
    safe_remove('tmp/airfoil.log')
    fname = 'tmp/airfoil.dat'
    with open(fname, 'wb') as f:
        np.savetxt(f, airfoil)
    
    try:
        # Has error: Floating point exception (core dumped)
        # This is the "empty input file: 'tmp/airfoil.log'" warning in other approaches
        child = pexpect.spawn('xfoil')
        timeout = 10
        
        child.expect('XFOIL   c> ', timeout)
        child.sendline('load tmp/airfoil.dat')
        child.expect('Enter airfoil name   s> ', timeout)
        child.sendline('af')
        child.expect('XFOIL   c> ', timeout)
        child.sendline('OPER')
        child.expect('.OPERi   c> ', timeout)
        child.sendline('VISC {}'.format(reynolds))
        child.expect('.OPERv   c> ', timeout)
        child.sendline('ITER {}'.format(n_iter))
        child.expect('.OPERv   c> ', timeout)
        child.sendline('MACH {}'.format(mach))
        child.expect('.OPERv   c> ', timeout)
        child.sendline('PACC')
        child.expect('Enter  polar save filename  OR  <return> for no file   s> ', timeout)
        child.sendline('tmp/airfoil.log')
        child.expect('Enter  polar dump filename  OR  <return> for no file   s> ', timeout)
        child.sendline()
        child.expect('.OPERva   c> ', timeout)
        child.sendline('ALFA {}'.format(alpha))
        child.expect('.OPERva   c> ', timeout)
        child.sendline()
        child.expect('XFOIL   c> ', timeout)
        child.sendline('quit')
        
        child.expect(pexpect.EOF)
        child.close()
        
        # Has the dead lock issue
#        with open('tmp/control.in', 'w') as text_file:
#            text_file.write('load tmp/airfoil.dat\n' +
#                            'af\n' +
#                            'OPER\n' +
#                            'VISC {}\n'.format(reynolds) +
#                            'ITER {}\n'.format(n_iter) +
#                            'MACH {}\n'.format(mach) +
#                            'PACC\n' +
#                            'tmp/airfoil.log\n' +
#                            '\n' +
#                            'ALFA {}\n'.format(alpha) +
#                            '\n' +
#                            'quit\n')
#        os.system('xfoil <tmp/control.in> tmp/airfoil.out')
        
        # Has the dead lock issue
        # Has memory issue
#        ps = sp.Popen(['xfoil'], stdin=sp.PIPE, stderr=sp.PIPE, stdout=sp.PIPE)
#        
#        # Use communicate() rather than .stdin.write, .stdout.read or .stderr.read 
#        # to avoid deadlocks due to any of the other OS pipe buffers filling up and 
#        # blocking the child process.
#        out, err = ps.communicate('load tmp/airfoil.dat\n' +
#                                  'af\n' +
#                                  'OPER\n' +
#                                  'VISC {}\n'.format(reynolds) +
#                                  'ITER {}\n'.format(n_iter) +
#                                  'MACH {}\n'.format(mach) +
#                                  'PACC\n' +
#                                  'tmp/airfoil.log\n' +
#                                  '\n' +
#                                  'ALFA {}\n'.format(alpha) +
#                                  '\n' +
#                                  'quit\n')
    
        res = np.loadtxt('tmp/airfoil.log', skiprows=12)
        if len(res) == 9:
            CL = res[1]
            CD = res[2]
        else:
            CL = -np.inf
            CD = np.inf
            
    except Exception as ex:
#        print(ex)
        print('XFoil error!')
        CL = -np.inf
        CD = np.inf
        
    safe_remove(':00.bl')
    
    return CL, CD

def evaluate(airfoil, return_CL_CD=False):
    # Airfoil operating conditions
    reynolds = 1.8e6
    mach = 0.01
    alpha = 0
    n_iter = 200
    
    CL, CD = compute_coeff(airfoil, reynolds, mach, alpha, n_iter)
    perf = CL/CD
        
    if np.isnan(perf) or perf > 300:
        perf = -1
    if return_CL_CD:
        return perf, CL, CD
    else:
        return perf
    
    
if __name__ == "__main__":
    
#    airfoil = np.load('tmp/a18sm.npy')
    airfoils = np.load('airfoil_interp.npy')
    airfoil = airfoils[np.random.choice(airfoils.shape[0])]
    
    reynolds = 1.8e6
    mach = 0.01
    alpha = 0
    n_iter = 200
    CL, CD = compute_coeff(airfoil, reynolds, mach, alpha, n_iter)
    print(CL, CD, CL/CD)
