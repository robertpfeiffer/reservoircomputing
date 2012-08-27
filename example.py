from liquid import *
import itertools
import shelve

if raw_input("run Mackey-Glass?[y/n] ")=="y":
    from mackey_glass  import mackey_glass
    print "MACKEY_GLASS"

    sim_length=1000
    u_input=[[1] for t in range(sim_length)]
    u_train=[[t] for t in itertools.islice(mackey_glass(beta=2.0,gamma=1.0,tau=2.0,n=9.65,x=1.0,dt=0.001),sim_length)]
    d = shelve.open("esn.shlv")^
    machine = d["mackeyglass"]
    d.close()

    print "training set created"
    train_in,train_target = list(run_all([(u_input,u_train)],machine))

    print "ESN finished"
    w_out = linear_regression(train_in,train_target)
    print "regression finished"

    print w_out,w_out.size

    u_in_test=[[1] for t in range(sim_length)]
    u_target_test=[[t] for t in itertools.islice(mackey_glass(beta=2,gamma=1,tau=2,n=9.65,x=0.5,dt=0.001),100)]

    print "test set created"

    print square_error(machine,w_out,[(u_in_test,u_target_test)])
    #print square_error(machine2,w_out2,[(u_in_test,u_target_test)])


if raw_input("run sine waves?[y/n] ")=="y":
    print "SINE WAVES"

    sim_length=100
    u_input=[[math.sin(t/100.0)] for t in range(sim_length)]
    u_train=[[math.sin(t/100.0 + 0.05)] for t in range(sim_length)]
    u_input2=[[math.sin(t/140.0)] for t in range(sim_length)]
    u_train2=[[math.sin(t/140.0 + 0.05)] for t in range(sim_length)]
    u_input3=[[math.sin(t/160.0)] for t in range(sim_length)]
    u_train3=[[math.sin(t/160.0 + 0.05)] for t in range(sim_length)]
    machine=FeedbackESN(1,10,1)

    print "training set created"
    train_in,train_target = list(run_all([(u_input,u_train),
                                      (u_input2,u_train2),
                                     (u_input3,u_train3)],machine))

    print "ESN finished"

    w_out = linear_regression(train_in,train_target)
    print "regression finished"

    u_in_test=[[math.sin(t/120.0)] for t in range(sim_length)]
    u_target_test=[[math.sin(t/120.0 + 0.05)] for t in range(sim_length)]

    print "test set created"

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

