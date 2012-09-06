from liquid import *
import itertools
import shelve
import numpy
import matplotlib
import matplotlib.pyplot as plt

def triangle(n):
    return n - int(n)

def squarewave(n):
    return int(n) % 2

def plot_ESN_run(machine,input,target=None,w_out=None,n_plot=1000,n_throwaway=0,jumpstart=0):
        if n_plot == 0:
            plot_in=list(input)
            if target:
                plot_target=list(target)
            n_plot=len(plot_in)
        else:
            plot_in=list(itertools.islice(input,n_plot+n_throwaway))
            if target:
                plot_target=list(itertools.islice(target,n_plot+n_throwaway))

        if jumpstart > 0:
            plot_run=list(machine.predict1(plot_in,w_out,plot_target[:jumpstart]))
        else:
            plot_run=list(machine.predict1(plot_in,w_out))
        plot_echo=[echo.ravel() for val,echo in plot_run]
        plot_out=[val for val,echo in plot_run]

        plot_in=numpy.array(plot_in[n_throwaway:]).T
        plot_echo=numpy.array(plot_echo[n_throwaway:]).T
        plot_out=numpy.array(plot_out[n_throwaway:]).T
        if target:
            plot_target=numpy.array(plot_target[n_throwaway:]).T

        plt.subplot(3,1,1)
	if plot_in.shape[0]>50:
         plt.pcolormesh(plot_in,cmap="bone")
        else:
         for d in range(plot_in.shape[0]):
            plt.plot(plot_in[d])

        plt.subplot(3,1,2)
        for d in range(plot_out.shape[0]):
            if target:
                plt.plot(plot_target[d])
            plt.plot(plot_out[d])
        if jumpstart > 0 and jumpstart-n_throwaway > 0:
            plt.axvline(jumpstart-n_throwaway)

        plt.subplot(3,1,3)
        plt.pcolormesh(plot_echo,cmap="bone")

        plt.matshow(machine.w_echo,cmap="bone")
        plt.show()

def plot_ESN_response(machine,input,n_throwaway=0):
        plot_in=list(input)
        n_plot=len(plot_in)
        w_out=numpy.ones((1,machine.nnodes+1))
        plot_run=list(machine.predict1(plot_in,w_out))
        plot_echo=[echo.ravel() for val,echo in plot_run]
        plot_out=[val for val,echo in plot_run]

        plot_in=numpy.array(plot_in[n_throwaway:]).T
        plot_echo=numpy.array(plot_echo[n_throwaway:]).T
        plot_out=numpy.array(plot_out[n_throwaway:]).T

        plt.subplot(3,1,1)
	if plot_in.shape[0]>50:
         plt.pcolormesh(plot_in,cmap="bone")
        else:
         for d in range(plot_in.shape[0]):
            plt.plot(plot_in[d])

        plt.subplot(3,1,2)
        for d in range(plot_out.shape[0]):
            plt.plot(plot_out[d])

        plt.subplot(3,1,3)
        plt.pcolormesh(plot_echo,cmap="bone")

        plt.matshow(machine.w_echo,cmap="bone")
        plt.show()

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


def wave_training_data_timescale(sim_length,timescale):
    pairs=[]
    for timescale in [timescale * 1.1, timescale * 0.9, timescale * 0.95, timescale * 1.05]:
        u_input=([math.sin(t/timescale)] for t in range(sim_length))
        u_target=([math.sin(0.75+t/timescale)] for t in range(sim_length))
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

def wave_test_data_timescale(sim_length,timescale):
    u_input=([math.sin(t/timescale)] for t in range(sim_length))
    u_target=([math.sin(0.75+t/timescale)] for t in range(sim_length))
    return u_input,u_target

def square_test_data_timescale(sim_length,timescale):
    u_input=([squarewave(t/timescale)] for t in range(sim_length))
    u_target=([squarewave(0.5+t/timescale)] for t in range(sim_length))
    return u_input,u_target

if raw_input("run Mackey-Glass?[y/n] ")=="y":
    from mackey_glass  import mackey_glass
    print "MACKEY_GLASS"

    sim_length=1000000
    dt=0.0001
    u_input=([1] for t in range(sim_length))
    u_train1=([t] for t in itertools.islice(mackey_glass(beta=2.0,gamma=1.0,tau=2.0,n=9.65,x=1.1,dt=dt),sim_length))
    u_train2=([t] for t in itertools.islice(mackey_glass(beta=2.0,gamma=1.0,tau=2.0,n=9.65,x=1.0,dt=dt),sim_length))
    u_train3=([t] for t in itertools.islice(mackey_glass(beta=2.0,gamma=1.0,tau=2.0,n=9.65,x=0.9,dt=dt),sim_length))
    u_train4=([t] for t in itertools.islice(mackey_glass(beta=2.0,gamma=1.0,tau=2.0,n=9.65,x=0.5,dt=dt),sim_length))
    u_train5=([t] for t in itertools.islice(mackey_glass(beta=2.0,gamma=1.0,tau=2.0,n=9.65,x=0.4,dt=dt),sim_length))
    u_train6=([t] for t in itertools.islice(mackey_glass(beta=2.0,gamma=1.0,tau=2.0,n=9.65,x=0.1,dt=dt),sim_length))
    u_train7=([t] for t in itertools.islice(mackey_glass(beta=2.0,gamma=1.0,tau=2.0,n=9.65,x=0.3,dt=dt),sim_length))
    u_train8=([t] for t in itertools.islice(mackey_glass(beta=2.0,gamma=1.0,tau=2.0,n=9.65,x=1.1,dt=dt),sim_length))
    u_train9=([t] for t in itertools.islice(mackey_glass(beta=2.0,gamma=1.0,tau=2.0,n=9.65,x=1.0,dt=dt),sim_length))
    u_train10=([t] for t in itertools.islice(mackey_glass(beta=2.0,gamma=1.0,tau=2.0,n=9.65,x=0.9,dt=dt),sim_length))
    u_train11=([t] for t in itertools.islice(mackey_glass(beta=2.0,gamma=1.0,tau=2.0,n=9.65,x=0.5,dt=dt),sim_length))
    u_train12=([t] for t in itertools.islice(mackey_glass(beta=2.0,gamma=1.0,tau=2.0,n=9.65,x=0.4,dt=dt),sim_length))
    u_train13=([t] for t in itertools.islice(mackey_glass(beta=2.0,gamma=1.0,tau=2.0,n=9.65,x=0.1,dt=dt),sim_length))
    u_train14=([t] for t in itertools.islice(mackey_glass(beta=2.0,gamma=1.0,tau=2.0,n=9.65,x=0.3,dt=dt),sim_length))
    # d = shelve.open("esn.shlv")
    # machine = d["mackeyglass"]
    # d.close()
    machine=DiagonalFeedbackESN(1,30,1)

    print "training set created"

    w_out= linear_regression_streaming([(u_input,u_train1),(u_input,u_train2),(u_input,u_train3),
                                        (u_input,u_train4),(u_input,u_train5),(u_input,u_train6),
					(u_input,u_train7),
					(u_input,u_train8),(u_input,u_train9),(u_input,u_train10),
                                        (u_input,u_train11),(u_input,u_train12),(u_input,u_train13),
					(u_input,u_train14)],machine)
    #print train_in,train_target
    print "ESN finished"
    #print w_out,w_out.size

    #w_out = linear_regression(train_in,train_target)
    print "regression finished"
    print w_out,w_out.size

    u_in_test=([1] for t in range(sim_length))
    u_target_test=([t] for t in itertools.islice(mackey_glass(beta=2,gamma=1,tau=2,n=9.65,x=0.55,dt=0.01),sim_length))

    print "test set created"

    plot_ESN_run(machine,u_in_test,u_target_test,w_out,10000,1000,3000)
    print square_error(machine,w_out,[(u_in_test,u_target_test)])


if raw_input("run sine waves?[y/n] ")=="y":
    print "SINE WAVES"

    sim_length=10000
    u_input=([math.sin(t/100.0)] for t in range(sim_length))
    u_train=([math.sin(t/100.0 + 0.75),math.sin(t/100.0 + 1.75)] for t in range(sim_length))
    u_input2=([math.sin(t/140.0)] for t in range(sim_length))
    u_train2=([math.sin(t/140.0 + 0.75),math.sin(t/140.0 + 1.75)] for t in range(sim_length))
    u_input3=([math.sin(t/160.0)] for t in range(sim_length))
    u_train3=([math.sin(t/160.0 + 0.75),math.sin(t/160.0 + 1.75)] for t in range(sim_length))

    machine=ESN(1,15)
    print "training set created"

    w_out= linear_regression_streaming([(u_input,u_train),
                                      (u_input2,u_train2),
                                     (u_input3,u_train3)],machine)

    print "ESN finished"
    print "regression finished"

    u_in_test=([math.sin(t/120.0)] for t in range(sim_length))
    u_target_test=([math.sin(t/120.0 + 0.75),math.sin(t/120.0 + 1.75)] for t in range(sim_length))

    print "test set created"

    plot_ESN_run(machine,u_in_test,u_target_test,w_out,1000,1000)
    print square_error(machine,w_out,[(u_in_test,u_target_test)])

if raw_input("run sine waves? (Diagonal)[y/n] ")=="y":
    print "SINE WAVES"

    sim_length=10000
    u_input=([math.sin(t/100.0)] for t in range(sim_length))
    u_train=([math.sin(t/100.0 + 0.75),math.sin(t/100.0 + 1.75)] for t in range(sim_length))
    u_input2=([math.sin(t/140.0)] for t in range(sim_length))
    u_train2=([math.sin(t/140.0 + 0.75),math.sin(t/140.0 + 1.75)] for t in range(sim_length))
    u_input3=([math.sin(t/160.0)] for t in range(sim_length))
    u_train3=([math.sin(t/160.0 + 0.75),math.sin(t/160.0 + 1.75)] for t in range(sim_length))

    machine=DiagonalESN(1,15)
    print "training set created"

    w_out= linear_regression_streaming([(u_input,u_train),
                                      (u_input2,u_train2),
                                     (u_input3,u_train3)],machine)

    print "ESN finished"
    print "regression finished"

    u_in_test=([math.sin(t/120.0)] for t in range(sim_length))
    u_target_test=([math.sin(t/120.0 + 0.75),math.sin(t/120.0 + 1.75)] for t in range(sim_length))

    print "test set created"

    plot_ESN_run(machine,u_in_test,u_target_test,w_out,1000,1000)
    print square_error(machine,w_out,[(u_in_test,u_target_test)])


if raw_input("run sine waves? (Bubble)[y/n] ")=="y":
    print "SINE WAVES"

    sim_length=10000
    u_input=([math.sin(t/100.0)] for t in range(sim_length))
    u_train=([math.sin(t/100.0 + 0.75),math.sin(t/100.0 + 1.75)] for t in range(sim_length))
    u_input2=([math.sin(t/140.0)] for t in range(sim_length))
    u_train2=([math.sin(t/140.0 + 0.75),math.sin(t/140.0 + 1.75)] for t in range(sim_length))
    u_input3=([math.sin(t/160.0)] for t in range(sim_length))
    u_train3=([math.sin(t/160.0 + 0.75),math.sin(t/160.0 + 1.75)] for t in range(sim_length))

    machine=BubbleESN(1,(10,10,10))
    print "training set created"

    w_out= linear_regression_streaming([(u_input,u_train),
                                      (u_input2,u_train2),
                                     (u_input3,u_train3)],machine)

    print "ESN finished"
    print "regression finished"

    u_in_test=([math.sin(t/120.0)] for t in range(sim_length))
    u_target_test=([math.sin(t/120.0 + 0.75),math.sin(t/120.0 + 1.75)] for t in range(sim_length))

    print "test set created"

    plot_ESN_run(machine,u_in_test,u_target_test,w_out,1000,1000)
    print square_error(machine,w_out,[(u_in_test,u_target_test)])

if raw_input("run sine waves? (Grid)[y/n] ")=="y":
    print "SINE WAVES"

    sim_length=10000
    u_input=([math.sin(t/100.0)] for t in range(sim_length))
    u_train=([math.sin(t/100.0 + 0.75),math.sin(t/100.0 + 1.75)] for t in range(sim_length))
    u_input2=([math.sin(t/140.0)] for t in range(sim_length))
    u_train2=([math.sin(t/140.0 + 0.75),math.sin(t/140.0 + 1.75)] for t in range(sim_length))
    u_input3=([math.sin(t/160.0)] for t in range(sim_length))
    u_train3=([math.sin(t/160.0 + 0.75),math.sin(t/160.0 + 1.75)] for t in range(sim_length))

    machine=Grid_3D_ESN(1,(10,10,10),4)
    print "training set created"

    w_out= linear_regression_streaming([(u_input,u_train),
                                      (u_input2,u_train2),
                                     (u_input3,u_train3)],machine)

    print "ESN finished"
    print "regression finished"

    u_in_test=([math.sin(t/120.0)] for t in range(sim_length))
    u_target_test=([math.sin(t/120.0 + 0.75),math.sin(t/120.0 + 1.75)] for t in range(sim_length))

    print "test set created"

    plot_ESN_run(machine,u_in_test,u_target_test,w_out,1000,1000)
    print square_error(machine,w_out,[(u_in_test,u_target_test)])

if raw_input("run sine waves? (Feedback)[y/n] ")=="y":
    print "SINE WAVES"

    sim_length=10000
    u_input=([math.sin(t/100.0)] for t in range(sim_length))
    u_train=([math.sin(t/100.0 + 0.75),math.sin(t/100.0 + 1.75)] for t in range(sim_length))
    u_input2=([math.sin(t/140.0)] for t in range(sim_length))
    u_train2=([math.sin(t/140.0 + 0.75),math.sin(t/140.0 + 1.75)] for t in range(sim_length))
    u_input3=([math.sin(t/160.0)] for t in range(sim_length))
    u_train3=([math.sin(t/160.0 + 0.75),math.sin(t/160.0 + 1.75)] for t in range(sim_length))
    machine=FeedbackESN(1,15,2)

    print "training set created"
    w_out= linear_regression_streaming([(u_input,u_train),
                                      (u_input2,u_train2),
                                     (u_input3,u_train3)],machine)

    print "ESN finished"
    print "regression finished"

    u_in_test=([math.sin(t/120.0)] for t in range(sim_length))
    u_target_test=([math.sin(t/120.0 + 0.75),math.sin(t/120.0 + 1.75)] for t in range(sim_length))

    print "test set created"

    plot_ESN_run(machine,u_in_test,u_target_test,w_out,2000,1000,1200)
    print square_error(machine,w_out,[(u_in_test,u_target_test)])

if raw_input("run sentiment analysis?[y/n] ")=="y":
    import sentiment

    print "SENTIMENT ANALYSIS"
    machine3=FeedbackESN(500,500,1)
    training_set=list(sentiment.gen_examples(500))
    print "training set created"
    train_in,train_target = run_all(training_set,machine3)
    print "ESN finished"
    w_out=linear_regression(train_in,train_target)
    print "regression finished"
    test_set=list(sentiment.gen_examples(35))
    print "test set created"
    print accuracy(machine3,w_out,test_set,5,0)

