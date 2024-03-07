from helper import *
import sys

x = generate_long_sequences("ACGT")
print(type(x[0][0]))
print(sys.getsizeof(x))
print(x)


y = generate_onehot_encoding("ACGT")
print(type(y[0][0]))
print(sys.getsizeof(y))
print(y)

y1 = torch.ByteTensor(y)
print(type(y1[0][0]))
print(sys.getsizeof(y1))
print(y1)

print(sys.getsizeof("ACGT"))

z = encodeLabel(1)
print(type(z[0]))
print(sys.getsizeof(z))
print(z)
