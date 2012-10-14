from liquid import *
import itertools
import shelve
import numpy
from plotting import *

def triangle(n):
    return n - int(n)

def squarewave(n):
    return int(n) % 2

def wave_training_data(sim_length):
    timescale=100.0
    u_input=([math.sin(t/timescale)] for t in range(sim_length))
    u_target=([math.sin(t/timescale + 0.75),math.sin(t/timescale + 1.75),squarewave(t/timescale/math.pi),1] for t in range(sim_length))

    timescale=140.0
    u_input2=([math.sin(t/timescale)] for t in range(sim_length))
    u_target2=([math.sin(t/timescale + 0.75),math.sin(t/timescale + 1.75),squarewave(t/timescale/math.pi),1] for t in range(sim_length))

    timescale=160.0
    u_input3=([math.sin(t/timescale)] for t in range(sim_length))
    u_target3=([math.sin(t/timescale + 0.75),math.sin(t/timescale + 1.75),squarewave(t/timescale/math.pi),1] for t in range(sim_length))

    timescale=180.0
    u_input4=([math.sin(t/timescale)] for t in range(sim_length))
    u_target4=([math.sin(t/timescale + 0.75),math.sin(t/timescale + 1.75),squarewave(t/timescale/math.pi),1] for t in range(sim_length))

    return [(u_input,u_target),(u_input2,u_target2),
            (u_input3,u_target3),(u_input4,u_target4)]


def pulse_data(length,offset,duration):
    u_input=([1 if offset < i < offset+duration else 0] for i in range(length))
    return u_input

def pulse_training_data(length,offset,offset2,duration):
    pairs=[]
    for j in range(10):
        delay=numpy.random.poisson(offset)
        u_input=([1 if delay+offset < i < delay+offset+duration else 0] for i in range(length))
        u_target=([1 if delay+offset2 < i < delay+offset2+duration else 0] for i in range(length))
        pairs.append((u_input,u_target))
    return pairs

def wave_training_data_hard(sim_length):
    pairs=[]
    for timescale in [100.0,140.0,160.0,180.0,80.0]:
    	u_input=([math.sin(t/timescale)] for t in range(sim_length))
    	u_target=([math.sin(0.75+t/(timescale/n**2))+m for n,m in [(1,3),(1.5,5),(2,7),(3,9)]] for t in range(sim_length))
        pairs.append((u_input,u_target))
    return pairs

def square_training_data_timescale(sim_length,timescale):
    pairs=[]
    for timescale in [timescale * 1.1, timescale * 0.9, timescale * 0.95, timescale * 1.05]:
        u_input=([squarewave(t/timescale)] for t in range(sim_length))
        u_target=([squarewave(0.5+t/timescale)] for t in range(sim_length))
        pairs.append((u_input,u_target))
    return pairs

def wave_test_data(sim_length):
    timescale=120.0
    u_input=([math.sin(t/timescale)] for t in range(sim_length))
    u_target=([math.sin(t/timescale + 0.75),math.sin(t/timescale + 1.75),squarewave(t/timescale/math.pi),1] for t in range(sim_length))

    return u_input,u_target

def wave_test_data_hard(sim_length):
    timescale=120.0
    u_input=([math.sin(t/timescale)] for t in range(sim_length))
    u_target=([math.sin(0.75+t/(timescale/n**2))+m for n,m in [(1,3),(1.5,5),(2,7),(3,9)]] for t in range(sim_length))
    return u_input,u_target


def square_test_data_timescale(sim_length,timescale):
    u_input=([squarewave(t/timescale)] for t in range(sim_length))
    u_target=([squarewave(0.5+t/timescale)] for t in range(sim_length))
    return u_input,u_target

if raw_input("run Mackey-Glass?[y/n] ")=="y":
    from mackey_glass  import mackey_glass
    print "MACKEY_GLASS"

    sim_length=10000
    dt=0.01
    tau=2.0
    gamma=1.0
    n=9.65
    beta=2.0
    u_input=[[1] for t in range(sim_length)]
    u_train1=([t] for t in itertools.islice(mackey_glass(beta=beta,gamma=gamma,tau=tau,n=n,x=1.1,dt=dt),sim_length))
    u_train2=([t] for t in itertools.islice(mackey_glass(beta=beta,gamma=gamma,tau=tau,n=n,x=1.0,dt=dt),sim_length))
    u_train3=([t] for t in itertools.islice(mackey_glass(beta=beta,gamma=gamma,tau=tau,n=n,x=0.9,dt=dt),sim_length))
    u_train4=([t] for t in itertools.islice(mackey_glass(beta=beta,gamma=gamma,tau=tau,n=n,x=0.5,dt=dt),sim_length))
    u_train5=([t] for t in itertools.islice(mackey_glass(beta=beta,gamma=gamma,tau=tau,n=n,x=0.4,dt=dt),sim_length))
    u_train6=([t] for t in itertools.islice(mackey_glass(beta=beta,gamma=gamma,tau=tau,n=n,x=0.1,dt=dt),sim_length))
    u_train7=([t] for t in itertools.islice(mackey_glass(beta=beta,gamma=gamma,tau=tau,n=n,x=0.3,dt=dt),sim_length))
    u_train8=([t] for t in itertools.islice(mackey_glass(beta=beta,gamma=gamma,tau=tau,n=n,x=1.1,dt=dt),sim_length))
    u_train9=([t] for t in itertools.islice(mackey_glass(beta=beta,gamma=gamma,tau=tau,n=n,x=1.0,dt=dt),sim_length))
    u_train10=([t] for t in itertools.islice(mackey_glass(beta=beta,gamma=gamma,tau=tau,n=n,x=0.9,dt=dt),sim_length))
    u_train11=([t] for t in itertools.islice(mackey_glass(beta=beta,gamma=gamma,tau=tau,n=n,x=0.5,dt=dt),sim_length))
    u_train12=([t] for t in itertools.islice(mackey_glass(beta=beta,gamma=gamma,tau=tau,n=n,x=0.4,dt=dt),sim_length))
    u_train13=([t] for t in itertools.islice(mackey_glass(beta=beta,gamma=gamma,tau=tau,n=n,x=0.1,dt=dt),sim_length))
    u_train14=([t] for t in itertools.islice(mackey_glass(beta=beta,gamma=gamma,tau=tau,n=n,x=0.3,dt=dt),sim_length))
    machine=DelayFeedbackESN(1,300,1,(15,50,90,140,210,300,350,500))

    w_out= linear_regression_streaming([(u_input,u_train1),(u_input,u_train2),(u_input,u_train3),
                                        (u_input,u_train4),(u_input,u_train5),(u_input,u_train6),
					(u_input,u_train7),
					(u_input,u_train8),(u_input,u_train9),(u_input,u_train10),
                                        (u_input,u_train11),(u_input,u_train12),(u_input,u_train13),
					(u_input,u_train14)],machine)
    print w_out,w_out.size

    u_in_test=([1] for t in range(sim_length))
    u_target_test=([t] for t in itertools.islice(mackey_glass(beta=beta,gamma=gamma,tau=tau,n=n,x=0.55,dt=dt),sim_length))

    plot_ESN_run(machine,u_in_test,u_target_test,w_out,10000,1000,3000)
    u_in_test=([1] for t in range(sim_length))
    u_target_test=([t] for t in itertools.islice(mackey_glass(beta=2,gamma=gamma,tau=tau,n=n,x=0.55,dt=dt),sim_length))

    print square_error(machine,w_out,[(u_in_test,u_target_test)])

esns = [(ESN,(1,15)),
        (DiagonalESN,(1,15)),
        (DiagonalESN,(1,50)),
        (BubbleESN,(1,(10,10,10),0.4,0.3)),
        (BubbleESN,(1,(10,10,10,10,10,10),0.4,0.3)),
        (Grid_3D_ESN,(1,(5,5,5),2)),
        (FeedbackESN,(1,15,4)),
        (DelayFeedbackESN,(1,15,4,(15,30,50,90)))]

if raw_input("plot ESN response?[y/n] ")=="y":
    gamma=better_sigmoid
    machine = ESN(1,30,frac_exc=0.1,gamma=gamma)
    input=pulse_data(300,100,50)
    plot_ESN_response(machine,input,20)

    machine = ESN(1,30,frac_exc=0.9,gamma=gamma)
    input=pulse_data(300,100,50)
    plot_ESN_response(machine,input,20)

    machine = ESN(1,30,frac_exc=0.5,gamma=gamma)
    input=pulse_data(300,100,50)
    plot_ESN_response(machine,input,20)

if raw_input("different ESNs?[y/n] ")=="y":
  for ESN1,params in esns:
    print ESN1,params
    if raw_input("run sine and square waves?[y/n] ")=="y":
        print "SQUARE WAVES"
        machine=ESN1(*params)
        training_data=wave_training_data(10000)
        w_out= linear_regression_streaming(training_data,machine)
        u_in_test,u_target_test=wave_test_data(3000)
        if ESN1.feedback:
            plot_ESN_run(machine,u_in_test,u_target_test,w_out,1000,100,500)
        else:
            plot_ESN_run(machine,u_in_test,u_target_test,w_out,1000,100)
        u_in_test,u_target_test=wave_test_data(10000)
        print square_error(machine,w_out,[(u_in_test,u_target_test)])

if raw_input("different ESNs, multiple frequencies?[y/n] ")=="y":
  for ESN1,params in esns:
    print ESN1,params
    if raw_input("run sine waves?[y/n] ")=="y":
        print "SINE WAVES"
        machine=ESN1(*params)
        training_data=wave_training_data_hard(10000)
        w_out= linear_regression_streaming(training_data,machine)
        u_in_test,u_target_test=wave_test_data_hard(3000)
        if ESN1.feedback:
            plot_ESN_run(machine,u_in_test,u_target_test,w_out,1000,100,500)
        else:
            plot_ESN_run(machine,u_in_test,u_target_test,w_out,1000,100)
        u_in_test,u_target_test=wave_test_data_hard(10000)
        print square_error(machine,w_out,[(u_in_test,u_target_test)])

if raw_input("different timescales, square input?[y/n] ")=="y":
  machine=ESN(1,15)
  timescales = [1.0,5.0,10.0,20.0,40.0,60.0,80.0,200.0,500.0,1000.0]
  for timescale in timescales:
        print "timescale",timescale
        training_data=square_training_data_timescale(10000,timescale)
        w_out= linear_regression_streaming(training_data,machine)
        u_in_test,u_target_test=square_test_data_timescale(3000,timescale)
        plot_ESN_run(machine,u_in_test,u_target_test,w_out,1000,100)
        u_in_test,u_target_test=square_test_data_timescale(10000,timescale)
        print square_error(machine,w_out,[(u_in_test,u_target_test)])

if raw_input("different timescales, square input, feedback?[y/n] ")=="y":
  machine=FeedbackESN(1,15,1)
  timescales = [1.0,5.0,10.0,20.0,40.0,60.0,80.0,200.0,500.0,1000.0]
  for timescale in timescales:
        print "timescale",timescale
        training_data=square_training_data_timescale(10000,timescale)
        w_out= linear_regression_streaming(training_data,machine)
        u_in_test,u_target_test=square_test_data_timescale(3000,timescale)
        plot_ESN_run(machine,u_in_test,u_target_test,w_out,1000,100)
        u_in_test,u_target_test=square_test_data_timescale(10000,timescale)
        print square_error(machine,w_out,[(u_in_test,u_target_test)])

if raw_input("different timescales, square input, delayed feedback?[y/n] ")=="y":
  machine=DelayFeedbackESN(1,15,4,(15,30,50,90))
  timescales = [1.0,5.0,10.0,20.0,40.0,60.0,80.0,200.0,500.0,1000.0]
  for timescale in timescales:
        print "timescale",timescale
        training_data=square_training_data_timescale(10000,timescale)
        w_out= linear_regression_streaming(training_data,machine)
        u_in_test,u_target_test=square_test_data_timescale(3000,timescale)
        plot_ESN_run(machine,u_in_test,u_target_test,w_out,1000,100)
        u_in_test,u_target_test=square_test_data_timescale(10000,timescale)
        print square_error(machine,w_out,[(u_in_test,u_target_test)])
