import numpy as np
import matplotlib.pyplot as plt


costFunct = []
theta0 = []
theta1 = []

y = np.array([0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1])
x = np.array([0.5,0.75,1.0,1.25,1.5,1.75,1.75,2.00,2.25,2.5,2.75,3.0,3.25,3.5,4.0,4.25,4.5,4.75,5.0,5.5])


def graph(formula, x_range):
    x = np.array(x_range)
    y = eval(formula)
    plt.plot(x, y)


#Implementing gradient descent
def calculateGradientDescent(x, y, theta, alpha, m, numIterations):
   xTrans = x.transpose()
   for i in range(0, numIterations):
       hypothesis = np.dot(x, theta)
       loss = hypothesis - y
       cost = np.sum(loss ** 2 ) / (2 * m) #
       costFunct.append(cost)
       gradient = np.dot(xTrans, loss) / m
       theta = theta - alpha * gradient
       theta0.append(theta[0])
       theta1.append(theta[1])
   return theta


# setting x0 as 1, inorder to generalize the equation
def mapXtoArray(numPoints, x1):
   x = np.zeros(shape=(numPoints, 2))
   for i in range(0, numPoints):
       x[i][0] = 1 #setting x0 as 1, inorder to generalize the equation
       x[i][1] = x1[i]
   return x


x = mapXtoArray(len(y), x)
m, n = np.shape(x)

numIterations= 2000
alpha = 0.1 #vary from 0.01, 0.1, 1
theta = np.ones(n) #initialize all the values of theta as 1
theta = calculateGradientDescent(x, y, theta, alpha, m, numIterations)


#Plotting values
plt.plot(x,y,'ro')
plt.ylabel('Y')
plt.xlabel('X')

s = str(theta[0]) + '+ x*' + str(theta[1])
print s


graph(s, range(0, 6))

plt.show()

plt.plot(theta0,costFunct)
plt.ylabel('Cost function J(theta0)')
plt.xlabel('theta0')
plt.show()

plt.plot(theta1,costFunct)
plt.ylabel('Cost function J(theta)')
plt.xlabel('theta1')
plt.show()

plt.plot(costFunct[1:500])
plt.ylabel('Cost function J(theta)')
plt.xlabel('Number of Iteration')
plt.show()

print("Theta value : %d",theta)

#[-0.15393527,  0.23459561]

