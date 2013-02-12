import numpy,math,random
import numpy.linalg as linalg
import itertools
import collections

from activations import *

class DummyESN(object):
    """This class implements the ESN interface, but it does not actually carry any
    state. Output = Input. Use it to compare the kernel quality against no kernel at all.
    Thus the regression is applied directly to the inputs."""
    feedback = False

    def __init__(self,ninput,nnodes,*a,**k):
        self.ninput=ninput
        self.nnodes=nnodes

    def run_streaming(self,u,y=None):
        return itertools.izip(u,y)

    def predict_with_echo(self,u,w_output):
        for ut in u:
            u_t=numpy.array(ut)
            state_1=numpy.append(u_t,numpy.ones(1))
            yield numpy.dot(w_output,state_1),u_t

    def predict(self,u,w_output):
      for y,x in self.predict_with_echo(u,w_output):
            yield y

def random_vector(size,a,b):
    return (b - a) * numpy.random.random_sample([size]) + a

def sech(x):
    return 1/numpy.cosh(x)

class ESN(object):
    feedback = False

    def __init__(self,input_dim,output_dim,leak_rate=1,conn_input=0.4,conn_recurrent=0.2,gamma=TanhActivation(),frac_exc=0.5, input_scaling=1, bias_scaling=1, spectral_radius=0.95, reset_state=True, start_in_equilibrium=True):
        self.ninput=input_dim
        self.nnodes=output_dim
        self.leak_rate=leak_rate
        self.gamma=gamma
        self.conn_recurrent=conn_recurrent
        self.conn_input=conn_input
        self.frac_exc=frac_exc
        self.reset_state = reset_state
        self.input_scaling = input_scaling
        self.bias_scaling = bias_scaling
        self.spectral_radius = spectral_radius
        self.start_in_equilibrium = start_in_equilibrium

        w_echo = numpy.array(
            [[self.connection_weight(i,j)
              for j in range(self.nnodes)]
            for i in range(self.nnodes)])
        w_input=numpy.array(
            [[self.input_weight(i,j)
              for j in range(self.ninput)]
            for i in range(self.nnodes)])
        w_add = numpy.array(
            [self.add_bias(i)
             for i in range(self.nnodes)])
        
        # set spectral radius of w_echo to ? (default = 0.95)
        eigenvalues=linalg.eigvals(w_echo)
        network_spectral_radius=max([abs(a) for a in eigenvalues])
        w_echo *= self.spectral_radius/network_spectral_radius
        
        self.w_echo = w_echo
        self.w_input = w_input
        self.w_add = w_add
        self.w_feedback = None
        self.current_feedback = None

        state1 = numpy.zeros(self.nnodes)
        zero_input=numpy.zeros(self.ninput)
        state2 = self.step(state1, zero_input)
        i = 0
        while not numpy.allclose(state1,state2):
            state1=state2
            state2=self.step(state1,zero_input)
            i +=1
            if i > 15000:
                break
        self.equilibrium_state = state2

        self.reset()
        
    def connection_weight(self,n1,n2):
        """recurrent synaptic strength for the connection from node n1 to node n2 """
        """ #Verteilung der Gewichte: Unform oder Gauss? Was bewirkt frac_exc?
        if random.random() < self.conn_recurrent:
            return numpy.random.rand()*2-1
        return 0
        """
        if random.random() < self.conn_recurrent:
            #weight = random.uniform(0,1)
            weight = random.gauss(0,1)
            #This stabilizes in the case of feedback. TODO: Investigate
            if random.uniform(0,1) < self.frac_exc:
                if weight <= 0:
                    weight=-weight
            else:
                if weight >= 0:
                    weight =-weight
            
            return weight
        return 0

    def input_weight(self,n1,n2):
        """synaptic strength for the connection from input node n1 to echo node n2"""
        if random.random() < self.conn_input:
            return random.uniform(-1, 1)*self.input_scaling
        return 0

    def add_bias(self,n1):
        """added to the neuron at each step,
        to make the neurons more different from each other"""
        return random.uniform(-1, 1) * self.bias_scaling

    def step(self, x_t_1, u_t, f_t=None):
        result = (1 - self.leak_rate) * x_t_1
        recur = numpy.dot(self.w_echo,x_t_1)
        inp   = numpy.dot(self.w_input,u_t)
        result += self.leak_rate * self.gamma.activate(
                   recur+inp+self.w_add)
        return result.ravel()

    def jacobian(self, u_t, x_t_1=None):
        if x_t_1 is None:
            state = self.current_state
        else:
            state = x_t_1
        recur = numpy.dot(self.w_echo,state)
        inp   = numpy.dot(self.w_input,u_t)
        J = numpy.dot(numpy.dot(sech(recur+inp+self.w_add),
                                sech(recur+inp+self.w_add).T),
                      numpy.dot(self.w_input,numpy.eye(u_t.size)))
        return J
    
    def run_batch(self, u, state=None):
        """ Runs the machine, returns the last state, saves previous states in state_echo
            Parameter u is the input, a 2dnumpy-array (time x input-dim) 
        """
        length = u.shape[0]
        if state is not None:
            pass
        elif not self.reset_state:
            state = self.current_state
        elif self.start_in_equilibrium:
            state = self.equilibrium_state
        else:
            state = numpy.zeros(self.nnodes)
        state_echo = numpy.zeros((length, self.nnodes))
        for i in range(length):
            u_t = u[i,:]
            state = self.step(state,u_t)
            state_echo[i,:] = state[:]
        self.current_state=state
        return state_echo

    def run_batch_feedback(self, u, state=None):
        """ Runs the machine, returns the last state, saves previous states in state_echo
            Parameter u is the input, a 2dnumpy-array (time x input-dim) 
        """
        length,inputs = u.shape
        state_echo = numpy.zeros((length, self.nnodes))
        if state is not None:
            pass
        elif not self.reset_state:
            state = self.current_state
        elif self.start_in_equilibrium:
            state = self.equilibrium_state
        else:
            state = numpy.zeros(self.nnodes)
        if self.w_feedback is not None and self.current_feedback is None:
            self.current_feedback = numpy.zeros(self.ninput-inputs)
        u_t=numpy.zeros(self.ninput)
        for i in range(length):
            u_t[:inputs] = u[i,:].ravel()
            if self.w_feedback is not None:
                u_t[inputs:] = self.current_feedback
            state    = self.step(state,u_t)
            state_echo[i,:] = state[:]
            if self.w_feedback is not None:
                state_1  = numpy.append(numpy.ones(1), state)
                self.current_feedback = numpy.dot(self.w_feedback.T,state_1)
        self.current_state = state
        return state_echo
    
    def reset(self):
        if self.start_in_equilibrium:
            self.current_state = self.equilibrium_state
        else:
            self.current_state = numpy.zeros(self.nnodes)

    def fold_in_feedback(self):
        pass

class Grid_3D_ESN(ESN):
    """In this ESN, the neurons are arranged in a 3D-grid.
    Connections only happen when the distance in the grid is smaller than a set threshold"""
    def connection_weight(self,n1,n2):
        """recurrent synaptic strength for the connection from node n1 to node n2"""
        x,y,z=self.dim
        x1=n1 % x
        x2=n2 % x
        y1=(n1 / x) % y
        y2=(n2 / x) % y
        z1=n1 / (x*y)
        z2=n2 / (x*y)
        p1=numpy.array([x1,y1,z1])
        p2=numpy.array([x2,y2,z2])
        dist=math.sqrt(numpy.inner(p1-p2,p1-p2))
        if dist < self.conn_length:
            if random.random() < self.conn_recurrent:
                return random.gauss(0,1)
        return 0

    def __init__(self,ninput,(x,y,z),max_length,*args,**kwargs):
        self.dim=(x,y,z)
        self.conn_length=max_length
        ESN.__init__(self,ninput,x*y*z,*args,**kwargs)
        self.ninput=ninput

BUBBLE_RATIO=5

class BubbleESN(ESN):
    """ESN with a n-bubble-architecture.
    The neurons in each bubble are densely connected.
    There are sparse connection from a bubble to later bubble_borders, but none back.
    """
    def bubble_index(self,n):
        k = 0
        for bubblemin,bubblemax in self.bubble_borders:
            if (bubblemin <= n and n < bubblemax):
                return k
            k += 1
        
    def connection_weight(self,n2,n1):
        """recurrent synaptic strength for the connection from node n1 to node n2"""
        n1_bubble = self.bubble_index(n1)
        n2_bubble = self.bubble_index(n2)

        if (n1_bubble == n2_bubble):
            if random.random() < self.conn_recurrent:
                return random.gauss(0,1)  
        if random.random() < self.conn_recurrent/BUBBLE_RATIO:
            return BUBBLE_RATIO * random.gauss(0,1)

        return 0

    def __init__(self,ninput,bubble_sizes,leak_rates=None,*args,**kwargs):
        self.bubble_borders=[]
        self.leak_rates = None
        s=0
        for bubble_size in bubble_sizes:
            min_=s
            max_=s+bubble_size
            s=max_
            self.bubble_borders.append((min_,max_))
        ESN.__init__(self,ninput,sum(bubble_sizes),*args,**kwargs)
        if self.leak_rates is not None:
            self.leak_rate=numpy.ones(self.nnodes)
            for (bmin,bmax),lr in zip(self.bubble_borders,leak_rates):
                self.leak_rate[bmin:bmax]=numpy.ones(bmax-bmin)*lr

class DecoupledBubbleESN(BubbleESN):
    def connection_weight(self,n2,n1):
        """recurrent synaptic strength for the connection from node n1 to node n2"""
        n1_bubble = self.bubble_index(n1)
        n2_bubble = self.bubble_index(n2)
        
        # weights if both neurons are in the same bubble
        if (n1_bubble == n2_bubble):
            if random.random() < self.conn_recurrent:
                return random.gauss(0,1)
        return 0

class FirstBubbleInput(BubbleESN):
    def input_weight(self,n2,n1):
        """synaptic strength for the connection from input node n1 to echo node n2"""
        if random.random() < self.conn_input:
            min_,max_=self.bubble_borders[0]
            if n2<max_:
                return random.uniform(-1, 1)*self.input_scaling
        return 0

class ForwardBubbleESN(FirstBubbleInput):
    def connection_weight(self,n2,n1):
        """recurrent synaptic strength for the connection from node n1 to node n2"""
        n1_bubble = self.bubble_index(n1)
        n2_bubble = self.bubble_index(n2)
        
        # weights if both neurons are in the same bubble
        if (n1_bubble == n2_bubble):
            if random.random() < self.conn_recurrent:
                return random.gauss(0,1)  
        # weights if the neurons are one bubble apart
        if (n1_bubble == n2_bubble-1):
            if random.random() < self.conn_recurrent/5:
                return random.gauss(0,1)
        return 0

class NeighbourBubbleESN(FirstBubbleInput):
    def connection_weight(self,n2,n1):
        """recurrent synaptic strength for the connection from node n1 to node n2"""
        n1_bubble = self.bubble_index(n1)
        n2_bubble = self.bubble_index(n2)
        
        # weights if both neurons are in the same bubble
        if (n1_bubble == n2_bubble):
            if random.random() < self.conn_recurrent:
                return random.gauss(0,1)  
        # weights if the neurons are one bubble apart
        if (n1_bubble == n2_bubble+1 or n1_bubble == n2_bubble-1):
            if random.random() < self.conn_recurrent/5:
                return random.gauss(0,1)
        return 0

class KitchenSinkBubbleESN(BubbleESN):
    """ESN with a n-bubble-architecture.
    The neurons in each bubble are densely connected.
    There are sparse connection from a bubble to later bubble_borders, but none back.
    """
        
    def connection_weight(self,n2,n1):
        """recurrent synaptic strength for the connection from node n1 to node n2"""
        n1_bubble = self.bubble_index(n1)
        n2_bubble = self.bubble_index(n2)

        if n1==n2:
            if self.diagonal_zero:
                return 0
        
        if (n1_bubble == n2_bubble):
            if random.random() < self.conn_recurrent:
                return random.gauss(0,1)
        if  self.interconnceted:
            if n2>n1:
                if n2_bubble==n1_bubble+1:
                    if random.random() < self.conn_recurrent/BUBBLE_RATIO:
                        return BUBBLE_RATIO * random.gauss(0,1)
                    return 0
                else:
                    if random.random() < self.conn_recurrent/BUBBLE_RATIO:
                        return BUBBLE_RATIO * random.gauss(0,1)
                    return 0
            elif n2<n1 and self.backward:
                if n2_bubble==n1_bubble+1:
                    if random.random() < self.conn_recurrent/BUBBLE_RATIO:
                        return BUBBLE_RATIO * random.gauss(0,1)
                    return 0
                else:
                    if random.random() < self.conn_recurrent/BUBBLE_RATIO:
                        return BUBBLE_RATIO * random.gauss(0,1)
                    return 0
        return 0

    def input_weight(self,n2,n1):
        """synaptic strength for the connection from input node n1 to echo node n2"""
        if self.bubble_index(n2)in self.input_bubbles:
            if random.random() < self.conn_input:
                return random.uniform(-1, 1)*self.input_scaling
        return 0

    def __init__(self,ninput,bubble_sizes,leak_rates=None,input_bubbles=[0],far=True,backward=True,interconnected=True,diagonal_zero=False,*args,**kwargs):
        self.input_bubbles=input_bubbles
        self.far=far
        self.backward=backward
        self.interconnected=interconnected
        self.diagonal_zero=diagonal_zero
        BubbleESN.__init__(self,ninput,bubble_sizes,leak_rates=leak_rates,*args,**kwargs)

def run_all(pairs,machine):
    inp = []
    targ = []
    for xs,ys in pairs:
        for xi,yi in machine.run_streaming(xs,ys):
            inp.append(xi)
            targ.append(yi)
    return numpy.vstack(inp),numpy.vstack(targ)

def square_error(machine,weights,testdata):
    n = 0
    m = 0
    err = 0.0
    for x,y in testdata:
        m += 1
        print "testing mse", m
        prediction=machine.predict(x,weights)
        for yc,yp in itertools.izip(y,prediction):
            n+=1
            err+=(yc-yp)**2
    return err/n

def accuracy(machine,weights,testdata,threshold,index):
    n = 0.0
    correct = 0.0
    avgcorrect = 0.0
    precision = 0.0
    nprecision = 0.0
    recall = 0.0
    nrecall = 0.0

    for x,y in testdata:
        n+=1
        print "testing accuracy", n

        prediction=machine.predict(x,weights)
        cursum=0.0
        curn=0.0
        for y,yp in itertools.izip(y,prediction):
            ylast=y
            plast=yp
            cursum+=yp[index]
            curn+=1
        if (cursum/curn > threshold) == (ylast[index] > threshold):
            avgcorrect+=1
        if (plast[index] > threshold) == (ylast[index] > threshold):
            correct+=1
        if (ylast[index] > threshold):
            # true positive + false negative
            nrecall+=1
            if (plast[index] > threshold):
                # true positive
                recall+=1
        if (plast[index] > threshold):
            # predicted positive
            nprecision+=1
            if (ylast[index] > threshold):
                # true positive
                precision+=1

    return correct/n, avgcorrect/n, precision/nprecision, recall/nrecall
