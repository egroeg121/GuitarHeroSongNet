def read(filename):
    import numpy as np

    #numpy datatypes in relation to the 'magic number'
    dtype_lib = {
            b'\x08': 'uint8',
            b'\x09': 'int8',
            b'\x0B': 'int16',
            b'\x0C': 'int32',
            b'\x0D': 'float32',
            b'\x0E': 'float64'}
    
    data=open(filename,'rb')

    #extract the datatype and no. of dimensions of the data from the 'magic number'
    data.seek(2)
    datatype=dtype_lib.get(data.read(1))
    no_dimensions=int.from_bytes(data.read(1),byteorder='big')

    #extract the dimensions of the data from the next bytes
    array_dimensions=[]
    for n in range(no_dimensions):
        array_dimensions.append(int.from_bytes(data.read(4),byteorder='big'))

    #create data array
    data_array=np.fromstring(data.read(),dtype=datatype).reshape(array_dimensions)
    return data_array