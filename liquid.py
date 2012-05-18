import numpy,math,random
import numpy.linalg as linalg

ninput=1
nnodes=50
sim_length=10000

# input: 1000 sinus values
u_input=[math.sin(t/100.0) for t in range(sim_length)]

u_train=[math.sin(t/100.0 + 0.05) for t in range(sim_length)]


#input_weights B (1x100 vector)
w_input=numpy.array([[1 for i in range(ninput)]
                    for j in range(nnodes)])

def connection_weight():
    # 1 in 5 Chance for a connection
    if random.randint(0,4)>0:
        return 0
    else:
        return random.gauss(0,1)

#reccurrent weights (A: echo layer matrix)
w_echo=numpy.array([[connection_weight() 
                     for j in range(nnodes)] 
                    for i in range(nnodes)])

#computes x(t), the echo state at time t
#x(t)=gamma(Ax(t-1)+Bu(t))
def step(gamma,x_t_1,u_t):
    return gamma(
        numpy.add(numpy.dot(w_echo,x_t_1),
                  w_input*u_t))

def run(u):
    # 1x100 matrix
    state = numpy.array([[0] for i in range(nnodes)])
    gamma=numpy.vectorize(math.tanh)
    for t in range(len(u)):
        state=step(gamma,state,u[t])
        yield state

state_rec=list(run(u_input))
a=numpy.transpose(state_rec[0])
for i in state_rec[1:]:
    a=numpy.append(a,numpy.transpose(i),axis=0)

a=numpy.append(a,numpy.array([[1] for i in range(sim_length)]),axis=1)

training_output=numpy.transpose(numpy.array([u_train]))
training_input = a

y=training_output
X=training_input
X_T=numpy.transpose(X)

w_output=numpy.dot(numpy.dot(linalg.pinv(numpy.dot(X_T,X)),X_T),y)
print w_output

error = 0
for i in range(sim_length):
    error+= ( training_output[i]-(numpy.dot(numpy.transpose(w_output[:(nnodes),:]),state_rec[i])+w_output[(nnodes),0])[0,0] ) ** 2

print error
print error ** 0.5
