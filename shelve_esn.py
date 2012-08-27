import liquid
import shelve
import sys
name = sys.argv[1]
inputs = int(sys.argv[2])
echo = int(sys.argv[3])
outputs = int(sys.argv[4])

machine=liquid.FeedbackESN(inputs,echo,outputs)
d = shelve.open("esn.shlv")
d[name]=machine

d.close()
