import numpy,math,random
import numpy.linalg as linalg
import itertools
import collections

def logistic(matrix):
    """logistic function, applied element-wise to a matrix.
    Can be used as a sigmoidal activation function"""
    return numpy.reciprocal(1+numpy.exp(numpy.negative(matrix)))

def smoothstep(matrix):
    """Can be used as a sigmoidal activation function"""
    clamped = numpy.clip(matrix,0.0,1.0)
    p1 = 3 * numpy.square(clamped)
    p2 = 2 * numpy.power(clamped,3*numpy.ones(clamped.shape))
    return p1-p2

def bad_sigmoid(matrix):
    """Can be used as a sigmoidal activation function"""
    return 1 +numpy.tanh(matrix)

def better_sigmoid(matrix):
    """logistic function, applied element-wise to a matrix.
    Can be used as a sigmoidal activation function"""
    return numpy.tanh(matrix)/2+smoothstep(matrix)

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

class ESN(object):
    feedback = False

    def connection_weight(self,n1,n2):
        """recurrent synaptic strength for the connection from node n1 to node n2"""
        if random.random() < self.conn_recurrent:
            weight = random.gauss(0,1)
            if random.uniform(0,1) < self.frac_exc:
                if weight > 0:
                   return weight
                else:
                   return -weight
            else:
                if weight < 0:
                   return weight
                else:
                   return -weight
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

    def __init__(self,ninput,nnodes,leak_rate=1,conn_input=0.4,conn_recurrent=0.2,gamma=numpy.tanh,frac_exc=0.5, input_scaling=1, bias_scaling=1, spectral_radius_scaling=0.95, reset_state=True, start_in_equilibrium=True):
        self.ninput=ninput
        self.nnodes=nnodes
        self.leak_rate=leak_rate
        self.gamma=gamma
        self.conn_recurrent=conn_recurrent
        self.conn_input=conn_input
        self.frac_exc=frac_exc
        self.reset_state = reset_state
        self.input_scaling = input_scaling
        self.bias_scaling = bias_scaling
        self.spectral_radius_scaling = spectral_radius_scaling

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
        
        # set spectral radius of w_echo to 0.95
        eigenvalues=linalg.eigvals(w_echo)
        spectral_radius=max([abs(a) for a in eigenvalues])
        w_echo *= self.spectral_radius_scaling/spectral_radius
        
        self.w_echo = w_echo
        self.w_input = w_input
        self.w_add = w_add

        state1 = numpy.zeros(self.nnodes)
        if start_in_equilibrium:
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
        else:
            self.equilibrium_state = state1
        self.current_state = self.equilibrium_state

    def step(self, x_t_1, u_t):
        result = (1-self.leak_rate)*x_t_1 + self.leak_rate*self.gamma(
                numpy.dot(self.w_echo,x_t_1)
             +  numpy.dot(self.w_input,u_t)
             +  self.w_add)
        return result.ravel()
    
    def run_batch(self, u):
        """ Runs the machine, returns the last state, saves previous states in state_echo
            Parameter u is the input, a 2dnumpy-array (time x input-dim) 
        """
        length = u.shape[0]
        if self.reset_state:
            state=self.equilibrium_state
        else:
            state = self.current_state
        
        state_echo = numpy.zeros((length, self.nnodes))
        for i in range(length):
            u_t = u[i,:]
            state = self.step(state,u_t)
            state_echo[i,:] = state[:]
        self.current_state = state        
        return state_echo
    
    def reset(self):
        self.current_state = self.equilibrium_state
        
    def run_streaming(self,u,y=None):
        """generate echo states and target values for training"""
        state=self.equilibrium_state
        for ut, yt in itertools.izip(u, y):
            u_t = numpy.array(ut)
            state = self.step(state,u_t)
            yield state, numpy.array(yt)


    def predict_with_echo(self,u,w_output):
        """generate echo states and predictions """
        state=self.equilibrium_state
        for ut in u:
            u_t=numpy.array(ut)
            state=self.step(state,u_t)
            state_1=numpy.append(state,numpy.ones(1))
            yield numpy.dot(w_output,state_1),state

    def predict(self,u,w_output):
      for y,x in self.predict_with_echo(u,w_output):
            yield y

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

    def input_weight(self,n1,n2):
        """synaptic strength for the connection from input node n1 to echo node n2"""
        if random.random() < self.conn_input:
            return 1
        return 0

    def __init__(self,ninput,(x,y,z),max_length,*args,**kwargs):
        self.dim=(x,y,z)
        self.conn_length=max_length
        ESN.__init__(self,ninput,x*y*z,*args,**kwargs)
        self.ninput=ninput

class BubbleESN(ESN):
    """ESN with a n-bubble-architecture.
    The neurons in each bubble are densely connected.
    There are sparse connection from a bubble to later bubbles, but none back.
    """
    def connection_weight(self,n1,n2):
        """recurrent synaptic strength for the connection from node n1 to node n2"""
        for bubblemin,bubblemax in self.bubbles:
            if (bubblemin <= n1 < bubblemax and
                bubblemin <= n2 < bubblemax):
                if random.random() < self.conn_recurrent:
                    return random.gauss(0,1)
        if (n1 < n2):
            if random.random() < self.conn_recurrent/5:
                return random.gauss(0,1)
        return 0

    def input_weight(self,n1,n2):
        """synaptic strength for the connection from input node n1 to echo node n2"""
        if random.random() < self.conn_input:
            min_,max_=self.bubbles[0]
            if n2<max_:
                return 1
        return 0

    def __init__(self,ninput,bubbles,*args,**kwargs):
        self.bubbles=[]
        s=0
        for b in bubbles:
            min_=s
            max_=s+b
            s=max_
            self.bubbles.append((min_,max_))
        ESN.__init__(self,ninput,sum(bubbles),*args,**kwargs)


class DiagonalESN(ESN):
    """ESN that is supposed to behave like an integrator"""
    def connection_weight(self,n1,n2):
        """recurrent synaptic strength for the connection from node n1 to node n2"""
        if n1==n2:
            return 1
        if random.random() < self.conn_recurrent:
            return random.gauss(0,0.1)
        return 0

class FeedbackESN(ESN):
    feedback = True

    def noise(self):
        return random_vector(self.noutput,-0.1,0.1)

    def __init__(self,ninput,nnodes,noutput,*args,**kwargs):
        ESN.__init__(self,ninput+noutput,nnodes,*args,**kwargs)
        self.ninput=ninput
        self.noutput=noutput
                
    def run_batch(self, u, y):
        """ Runs the machine, returns the last state, saves previous states in state_echo
            Parameter u is the input, a 2dnumpy-array (time x input-dim) 
        """
        length = u.shape[0]
        state=self.equilibrium_state
        state_echo = numpy.zeros((length, self.nnodes))
        i = 0
        for state, fb in self.run_streaming(state,y):
            state_echo[i,:] = state
            i = i + 1        
        return state_echo
    
    def run_streaming(self,x,y):
        state = self.equilibrium_state
        t = 0
        for xt,yt in itertools.izip(x,y):
            if t == 0:
                feedback = numpy.zeros(len(yt))
            u_t = numpy.append(
                numpy.array(xt),
                feedback+self.noise())
            state=self.step(state,u_t)
            feedback = numpy.array(yt)
            yield state, feedback
            t += 1

    def predict_with_echo(self,x,w_output,initial_feedback=[]):
        state = self.equilibrium_state
        feedback = numpy.zeros(self.noutput)
        l = len(initial_feedback)
        t = 0
        for xt in x:
            if t < l:
                feedback=numpy.array(initial_feedback[t])
            u_t = numpy.append(
                numpy.array(xt),
                feedback)
            state=self.step(u_t)
            state_1= numpy.append(self.state,numpy.ones(1))
            feedback = numpy.dot(w_output,state_1)
            yield feedback,state
            t += 1

class DelayFeedbackESN(ESN):
    feedback = True

    def noise(self):
        return random_vector(self.nfeedback,-0.1,0.1)

    def __init__(self,ninput,nnodes,noutput,delays,*args,**kwargs):
        self.nfeedback=noutput*len(delays)
        self.delays=delays
        self.maxdelay=max(*delays)+2
        ESN.__init__(self,ninput+noutput*len(delays),nnodes,*args,**kwargs)
        self.ninput=ninput
        self.noutput=noutput
    
    def run_streaming(self,x,y):
        state = self.equilibrium_state
        memory = collections.deque([],maxlen=self.maxdelay)
        feedback = numpy.zeros(self.nfeedback)
        d = zip(self.delays,range(len(self.delays)))
        for xt,yt in itertools.izip(x,y):
            target = numpy.array(yt)
            for delay,i in d:
                if delay < len(memory):
                    feedback[i*self.noutput:(i+1)*self.noutput]=memory[delay]
            u_t = numpy.append(
                numpy.array(xt),
                feedback+self.noise())
            state=self.step(state,u_t)
            memory.append(target)
            yield state, target

    def predict_with_echo(self,x,w_output,initial_feedback=[]):
        state = self.equilibrium_state
        memory = collections.deque([],maxlen=self.maxdelay)
        feedback = numpy.zeros(self.nfeedback)
        l = len(initial_feedback)
        t=0
        d = zip(self.delays,range(len(self.delays)))
        for xt in x:
            for delay,i in d:
                if delay < len(memory):
                    feedback[i*self.noutput:(i+1)*self.noutput]=memory[delay]
            u_t = numpy.append(
                numpy.array(xt),
                feedback)
            state=self.step(state,u_t)
            state_1= numpy.append(state,numpy.ones(1))
            value = numpy.dot(w_output,state_1)
            if t < l:
                memory.append(numpy.array(initial_feedback[t]))
            else:
                memory.append(value)

            yield value,state
            t += 1


class DiagonalFeedbackESN(FeedbackESN):
     def connection_weight(self,n1,n2):
         """recurrent synaptic strength for the connection from node n1 to node n2"""
         if n1==n2:
             return 1
         if random.random() < self.conn_recurrent:
             return random.gauss(0,0.4)
         return 0

def run_all(pairs,machine):
    inp = []
    targ = []
    for xs,ys in pairs:
        for xi,yi in machine.run_streaming(xs,ys):
            inp.append(xi)
            targ.append(yi)
    return numpy.vstack(inp),numpy.vstack(targ)

def linear_regression(X,Y):
    X=numpy.append(X,numpy.ones((X.shape[0],1)),axis=1)
    return numpy.transpose(linalg.lstsq(X,Y)[0])

def rprop(A,b):
         eta = numpy.ones((b.shape[1],A.shape[1]))*0.1
         w   = numpy.ones((b.shape[1],A.shape[1]))*numpy.std(b)/numpy.std(A)
         diff = numpy.dot((numpy.dot(A,w.T) - b).T,A)
         print A,b
         while numpy.sum(abs(diff)) > 0.1:
             w     = w - (eta*diff)
             diff2 = numpy.dot((numpy.dot(A,w.T) - b).T,A)
             eta   = eta - eta * 0.5 * (diff2*diff < 0) + eta * 0.2 * (diff2*diff > 0)
             diff  = diff2
         return w

def linear_regression_streaming(pairs,machine):
    """input parameters:
    pairs: pairs of lists of inputs and outputs
    machine: an ESN
    output:
    output weight matrix"""
    A = None 
    b = None 
    n = 0
    for xs,ys in pairs:
        n += 1
        print "training run_streaming", n
        for xi,yi in machine.run_streaming(xs,ys):
            xi=numpy.append(xi,numpy.ones(1))
            XTX=numpy.outer(xi,xi)
            XTy=numpy.outer(xi,yi)
            if A==None:
                assert b==None
                A=XTX
                b=XTy
            else:
                A = numpy.add(XTX,A)
                b = numpy.add(XTy,b)
    #return numpy.linalg.lstsq(A,b)[0].T
    return numpy.dot(linalg.pinv(A),b).T
    #return rprop(A,b)

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
