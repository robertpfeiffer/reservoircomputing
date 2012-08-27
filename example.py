from liquid import *
import itertools
import shelve
import numpy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

if raw_input("run Mackey-Glass?[y/n] ")=="y":
    from mackey_glass  import mackey_glass
    print "MACKEY_GLASS"

    sim_length=100000
    u_input=[[1] for t in range(sim_length)]
    u_train1=[[t] for t in itertools.islice(mackey_glass(beta=2.0,gamma=1.0,tau=2.0,n=9.65,x=1.1,dt=0.01),sim_length)]
    u_train2=[[t] for t in itertools.islice(mackey_glass(beta=2.0,gamma=1.0,tau=2.0,n=9.65,x=1.0,dt=0.01),sim_length)]
    u_train3=[[t] for t in itertools.islice(mackey_glass(beta=2.0,gamma=1.0,tau=2.0,n=9.65,x=0.9,dt=0.01),sim_length)]
    u_train4=[[t] for t in itertools.islice(mackey_glass(beta=2.0,gamma=1.0,tau=2.0,n=9.65,x=0.5,dt=0.01),sim_length)]
    u_train5=[[t] for t in itertools.islice(mackey_glass(beta=2.0,gamma=1.0,tau=2.0,n=9.65,x=0.4,dt=0.01),sim_length)]
    u_train6=[[t] for t in itertools.islice(mackey_glass(beta=2.0,gamma=1.0,tau=2.0,n=9.65,x=0.1,dt=0.01),sim_length)]
    u_train7=[[t] for t in itertools.islice(mackey_glass(beta=2.0,gamma=1.0,tau=2.0,n=9.65,x=0.3,dt=0.01),sim_length)]
    d = shelve.open("esn.shlv")
    machine = d["mackeyglass"]
    d.close()

    print "training set created"
    train_in,train_target = list(run_all([(u_input,u_train1),(u_input,u_train2),(u_input,u_train3),
                                          (u_input,u_train4),(u_input,u_train5),(u_input,u_train6),(u_input,u_train7)],machine))

    print "ESN finished"
    w_out = linear_regression(train_in,train_target)
    print "regression finished"

    #print w_out,w_out.size

    u_in_test=[[1] for t in range(sim_length)]
    u_target_test=[[t] for t in itertools.islice(mackey_glass(beta=2,gamma=1,tau=2,n=9.65,x=0.55,dt=0.01),sim_length)]

    print "test set created"

    n_plot=10000
    echo_states=numpy.array([a.ravel() for a in machine.run(u_in_test[:n_plot],u_target_test[n_plot:2*n_plot])]).T
    outputs=numpy.array(list(machine.predict(u_in_test[n_plot:2*n_plot],w_out))).ravel()
    input_function=numpy.array([u_in_test[n_plot:2*n_plot]]).ravel()
    target_function=numpy.array([u_target_test[n_plot:2*n_plot]]).ravel()

    plt.subplot(3,1,1)
    plt.plot(input_function)
    plt.subplot(3,1,2)
    plt.plot(target_function)
    plt.plot(outputs)
    plt.subplot(3,1,3)
    plt.pcolormesh(echo_states,cmap="bone")
    plt.show()
    #plt.bone()

    print square_error(machine,w_out,[(u_in_test,u_target_test)])
    #print square_error(machine2,w_out2,[(u_in_test,u_target_test)])


if raw_input("run sine waves?[y/n] ")=="y":
    print "SINE WAVES"

    sim_length=100000
    u_input=[[math.sin(t/100.0)] for t in range(sim_length)]
    u_train=[[math.sin(t/100.0 + 0.75)] for t in range(sim_length)]
    u_input2=[[math.sin(t/140.0)] for t in range(sim_length)]
    u_train2=[[math.sin(t/140.0 + 0.75)] for t in range(sim_length)]
    u_input3=[[math.sin(t/160.0)] for t in range(sim_length)]
    u_train3=[[math.sin(t/160.0 + 0.75)] for t in range(sim_length)]
    machine=FeedbackESN(1,10,1)
    machine=ESN(1,30)

    print "training set created"
    train_in,train_target = list(run_all([(u_input,u_train),
                                      (u_input2,u_train2),
                                     (u_input3,u_train3)],machine))

    print "ESN finished"

    w_out = linear_regression(train_in,train_target)
    print "regression finished"

    u_in_test=[[math.sin(t/120.0)] for t in range(sim_length)]
    u_target_test=[[math.sin(t/120.0 + 0.75)] for t in range(sim_length)]

    print "test set created"

    n_plot=1000
    echo_states=numpy.array([a.ravel() for a in machine.run(u_in_test[:n_plot],u_target_test[n_plot:2*n_plot])]).T
    outputs=numpy.array(list(machine.predict(u_in_test[n_plot:2*n_plot],w_out))).ravel()
    input_function=numpy.array([u_in_test[n_plot:2*n_plot]]).ravel()
    target_function=numpy.array([u_target_test[n_plot:2*n_plot]]).ravel()

    plt.subplot(3,1,1)
    plt.plot(input_function)
    plt.subplot(3,1,2)
    plt.plot(target_function)
    plt.plot(outputs)
    plt.subplot(3,1,3)
    plt.pcolormesh(echo_states,cmap="bone")
    plt.show()
    #plt.bone()

    print square_error(machine,w_out,[(u_in_test,u_target_test)])
    #print square_error(machine2,w_out2,[(u_in_test,u_target_test)])

    print square_error(machine,w_out,[(u_in_test,u_target_test)])

if raw_input("run sine waves? (Feedback)[y/n] ")=="y":
    print "SINE WAVES"

    sim_length=100000
    u_input=[[math.sin(t/100.0)] for t in range(sim_length)]
    u_train=[[math.sin(t/100.0 + 0.75)] for t in range(sim_length)]
    u_input2=[[math.sin(t/140.0)] for t in range(sim_length)]
    u_train2=[[math.sin(t/140.0 + 0.75)] for t in range(sim_length)]
    u_input3=[[math.sin(t/160.0)] for t in range(sim_length)]
    u_train3=[[math.sin(t/160.0 + 0.75)] for t in range(sim_length)]
    machine=FeedbackESN(1,10,1)

    print "training set created"
    train_in,train_target = list(run_all([(u_input,u_train),
                                      (u_input2,u_train2),
                                     (u_input3,u_train3)],machine))

    print "ESN finished"

    w_out = linear_regression(train_in,train_target)
    print "regression finished"

    u_in_test=[[math.sin(t/120.0)] for t in range(sim_length)]
    u_target_test=[[math.sin(t/120.0 + 0.75)] for t in range(sim_length)]

    print "test set created"

    n_plot=1000
    echo_states=numpy.array([a.ravel() for a in machine.run(u_in_test[:n_plot],u_target_test[n_plot:2*n_plot])]).T
    outputs=numpy.array(list(machine.predict(u_in_test[n_plot:2*n_plot],w_out))).ravel()
    input_function=numpy.array([u_in_test[n_plot:2*n_plot]]).ravel()
    target_function=numpy.array([u_target_test[n_plot:2*n_plot]]).ravel()

    plt.subplot(3,1,1)
    plt.plot(input_function)
    plt.subplot(3,1,2)
    plt.plot(target_function)
    plt.plot(outputs)
    plt.subplot(3,1,3)
    plt.pcolormesh(echo_states,cmap="bone")
    plt.show()
    #plt.bone()

    print square_error(machine,w_out,[(u_in_test,u_target_test)])
    #print square_error(machine2,w_out2,[(u_in_test,u_target_test)])

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

