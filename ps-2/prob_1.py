# PS2 -- Dario Loprete -- dl5521@nyu.edu
# PROBLEM 1

# imported packages
import numpy as np

x = 100.98763

# definition of function that converts the number from decimal to 32-bit representation
def get_bits(number):
    bytes = number.tobytes()
    bits = []
    for byte in bytes:
        bits = bits + np.flip(np.unpackbits(np.uint8(byte)), np.uint8(0)).tolist()
    return list(reversed(bits))

# definition of function that prints the 32-bit representation of the number
def print_bits(list):
    print("sign -> {0}\nexponent -> {1}\nmantissa -> {2}\n ".format(list[0], list[1:9], list[9:32]))

# definition of function that converts the 32-bit representation to decimal 
def float_to_dec(list):
   sign = list[0]
   exponent = list[1:9]
   mantissa = list[9:32]

   e_value = 0
   for index in range(len(exponent)):
    e_value += np.flip(exponent)[index]*2**(index)

   sum_mantissa = 0
   for i in range(1,24):
     sum_mantissa += np.flip(mantissa)[23 -i] * 2**(-i)

   return (-1)**sign * 2**(e_value - 127) * (1 + sum_mantissa)


# list which contains the bits for the 32-bit representation of the number
bitlist = get_bits(np.float32(x))

# printing of results
print("Inserted number:", x)
print("\nNumber in 32-bit representation:")
print_bits(bitlist)
print("Conversion from 32-bit representation to decimal:",float_to_dec(bitlist))
print("\nDifference: {0}\n".format(np.float64(np.abs(float_to_dec(bitlist)-x))))

