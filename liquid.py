import numpy,math,random
import numpy.linalg as linalg

def connection_weight():
    # 1 in 5 Chance for a connection
    if random.randint(0,4)>0:
        return 0
    else:
        return random.gauss(0,1)

class ESN(object):
    def connection_weight(self,n1,n2):
        return connection_weight()

    def input_weight(self,n1,n2):
        return 1 + random.gauss(0,1)

    def __init__(self,ninput,nnodes):
        self.ninput=ninput
        self.nnodes=nnodes
        self.gamma=numpy.vectorize(math.tanh)
        w_echo = [[self.connection_weight(i,j) 
                   for j in range(self.nnodes)] 
                  for i in range(self.nnodes)]
        self.w_echo = numpy.array(w_echo)
        w_input=[[self.input_weight(i,j)
                  for j in range(self.ninput)]
                 for i in range(self.nnodes)]
        self.w_input = numpy.array(w_input)
    
    def step(self,gamma,x_t_1,u_t):
        result = gamma(
            numpy.add(numpy.dot(self.w_echo,x_t_1),
                      numpy.dot(self.w_input,u_t)))
        return result

    def run(self,u,y=None):
        # initialize state to 0
        state = numpy.array([[0] for i in 
                             range(self.nnodes)])
        for t in range(len(u)):
            u_t=numpy.transpose(numpy.array([u[t]]))
            state=self.step(self.gamma,state,u_t)
            yield numpy.transpose(state)

    def predict(self,u,w_output):
        state = numpy.array([[0] for i in 
                             range(self.nnodes)])
        for t in range(len(u)):
            u_t=numpy.transpose(numpy.array([u[t]]))
            state=self.step(self.gamma,state,u_t)
            state_1=numpy.append(state,numpy.ones((1,1)),axis=0)
            yield numpy.dot(w_output,state_1)

class FeedbackESN(ESN):
    def connection_weight(self,n1,n2):
        return connection_weight()

    def input_weight(self,n1,n2):
        if self.ninput > 1:
            return 1+connection_weight()
        return 1

    def noise(self):
        return random.uniform(-0.2,0.2)

    def __init__(self,ninput,nnodes,noutput):
        self.ninput=ninput
        self.nnodes=nnodes
        self.noutput=noutput
        self.gamma=numpy.vectorize(math.tanh)
        w_echo = [[self.connection_weight(i,j) 
                   for j in range(self.nnodes)] 
                  for i in range(self.nnodes)]
        self.w_echo = numpy.array(w_echo)
        w_input=[[self.input_weight(i,j)
                  for j in range(self.ninput+
                                 self.noutput)]
                 for i in range(self.nnodes)]
        self.w_input = numpy.array(w_input)

    def run_noisy_feedback(self,x,y):
        state = numpy.array([[0] for i in 
                             range(self.nnodes)])
        for t in range(len(x)):
            x_t =numpy.transpose(numpy.array(
                   [x[t]+
                    ([yn+self.noise()
                      for yn in y[t-1]] if t>0 
                     else [0 for yn in y[0]])]))

            state=self.step(self.gamma,state,x_t)
            yield numpy.transpose(state)
    
    def run(self,x,y):
        return self.run_noisy_feedback(x,y)

    def predict(self,x,w_output):
        state = numpy.array([[0] for i in 
                             range(self.nnodes)])
        feedback = numpy.array([[0] for i in 
                             range(self.noutput)])
        for t in range(len(x)):
            x_t = numpy.vstack([
                numpy.transpose(numpy.array(
                   [x[t]])),
                feedback])
            state=self.step(self.gamma,state,x_t)
            state_1= numpy.append(state,numpy.ones((1,1)),axis=0)
            feedback = numpy.dot(w_output,state_1)
            yield feedback

def run_all(trainingdata,machine):
    def helper1():
        for (x,y) in trainingdata:
            for output in machine.run(x,y):
                yield output
    
    def helper2():
        for (x,y) in trainingdata:
            for l in y:
                yield numpy.array(l)

    return numpy.vstack(helper1()),numpy.vstack(helper2())

def linear_regression(X,Y):
    X=numpy.append(X,numpy.ones((X.shape[0],1)),axis=1)
    return numpy.transpose(linalg.lstsq(X,Y)[0])

def square_error(machine,weights,testdata):
    n = 0
    err = 0.0
    for x,y in testdata:
        prediction=machine.predict(x,weights)
        for x,y,yp in zip(x,y,prediction):
            n+=1
            err+=(y-yp)**2
    return err/n

def accuracy(machine,weights,testdata,threshold,index):
    n = 0.0
    correct = 0.0
    for x,y in testdata:
        prediction=machine.predict(x,weights)
        for x,y,yp in zip(x,y,prediction):
            n+=1
            if (yp[index] > threshold) == (y[index] > threshold):
                correct+=1
    return correct/n
