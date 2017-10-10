import numpy.linalg


alpha = 0.01

def findE(x, b):  # Function to find the matrix "E"
    print "X transpose : " , numpy.array(x).T
    f = numpy.dot(numpy.array(x).T,x)
    print "X transpose * x : ", f
    x_inv = numpy.linalg.pinv(f)  # Find the Moore-Penrose Inverse of x
    print "X transpose * x inverse : " ,x_inv
    g = numpy.dot(x_inv, numpy.array(x).T)
    print "Pseudo matrix",g
    print "x inverser : " , x_inv

    W = numpy.dot(g, b)  # Multiple x# and b

    print "W : ", W
    E = numpy.subtract(numpy.dot(x, W), b)  # Subtract b from x times W
    E_rounded = numpy.around(E.astype(numpy.double), 1)  # Round the E matrix to one decimal point

    print ("ERROR :" , E)


    b = numpy.add(b, alpha * numpy.add(E_rounded, numpy.absolute(E_rounded)))  # Add b to the addition of E and the absolute value of E
    print ("Value of b: " , b)

    print ("Theta : ", W)
    print("\n\n")
    return E_rounded, W


def checkEEZ(E):  # Check if the matrix "E" is equal to zero
    for row in E:
        if (row != 0):
            return False
    return True


def checkELZ(E):  # Check if the matrix "E" is less than zero
    for row in E:
        if (row > 0 or row == 0):
            return False
    return True


def done(W):  # Print out the equation of the line
    print "Equation of the line is x =", ((W[2] * -1) / W[0])  # Print statement for 2D problems
    # print "Equation of the line is x - y + z =", ((W[3]*-1)/W[0]) # Print statement for 3D problem


#x = [[0,0,1], [0,1,1], [-1,0,-1], [-1,-1,-1]] # Example one, solution is possible

x = [[1, 2, 4], [1, 3, 3], [-1, -6, -12], [-1, -8, -10]]  # Example two, solution is not possible
# x = [[0,0,1,1], [1,0,0,1], [1,0,1,1], [1,1,1,1],
#     [0,0,0,-1], [0,-1,0,-1], [0,-1,-1,-1], [-1,-1,0,-1]] # Example three, 3D problem, solution is a plane
b = [1, 1, 1, 1]
theta = [0, 1, 1]

while (True):  # Keep running until we find a solution or a solution is not possible
    E, theta = findE(x, b)
    alpha = 0.1
    EEZ = checkEEZ(E)
    ELZ = checkELZ(E)

    if (EEZ):
        done(theta)
        break
    elif (ELZ):
        print "No solution is possible"
        break
