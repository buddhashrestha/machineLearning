import numpy as np
import matplotlib.pyplot as plt

costFunct = []
thetaA0 = []
thetaA1 = []

y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
x = np.array(
    [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 1.75, 2.00, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 4.0, 4.25, 4.5, 4.75, 5.0, 5.5])


def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))


# m -> no. of sample data, x -> sigmoid functions output, y->output (either 1/0)
def costForLogistic(m, x, y):
    cost = -1 / m * np.sum(y * np.log(x) + (1 - y) * (np.log(1 - x)))
    return cost

def linearCostFunc(m, x,y):
    cost = 100/m * np.sum( y - 2 * y * sigmoid(x) + sigmoid(x))
    return cost

def linearGradientLoss(m,x,y):
    loss = 1/m * np.sum( sigmoid(x)*(1- sigmoid(x))*(1-2*y)*x)
    return loss


# Implementing gradient descent
def calculateGradientDescent(x, y, theta, alpha, m, numIterations):
    xTrans = x.transpose()
    theta0 =1
    theta1 =1
    for i in range(0, numIterations):
        xDotTheta0 = xTrans[0] * theta0
        xDotTheta1 = xTrans[1] * theta1

        hypothesis0 = sigmoid(xDotTheta0)
        hypothesis1 = sigmoid(xDotTheta1)

#        loss = hypothesis - y
        cost0 = linearCostFunc(m, hypothesis0 , y)
        cost1 = linearCostFunc(m, hypothesis1 , y)
        cost = np.array([cost0,cost1])

        gradient0 = linearGradientLoss(m,hypothesis0,y)
        gradient1 = linearGradientLoss(m, hypothesis1, y)
        costFunct.append(cost)

        theta0 = theta0 - alpha * gradient0
        theta1 = theta1 - alpha * gradient1

        thetaA0.append(theta0)
        thetaA1.append(theta1)
    return theta


# setting x0 as 1, inorder to generalize the equation
def mapXtoArray(numPoints, x1):
    x = np.zeros(shape=(numPoints, 2))
    for i in range(0, numPoints):
        x[i][0] = 1  # setting x0 as 1, inorder to generalize the equation
        x[i][1] = x1[i]
    return x


vectored_x = mapXtoArray(len(y), x)
m, n = np.shape(vectored_x)
print "m :"
print m

numIterations = 100000
alpha = 0.01  # vary from 0.01, 0.1, 1
theta = np.ones(n)  # initialize all the values of theta as 1
theta = calculateGradientDescent(vectored_x, y, theta, alpha, m, numIterations)

# Plotting values
plt.plot(vectored_x, y, 'ro')
plt.ylabel('Y')
plt.xlabel('X')

xDotTheta = np.dot(vectored_x, theta)
hypothesis = sigmoid(xDotTheta)

print hypothesis
plt.plot(x,hypothesis)
plt.ylabel("Sigmoid function")
plt.xlabel('Range of x')
#plt.show()
#graph(s, range(0, 6))

plt.show()

plt.plot(thetaA0, costFunct)
plt.ylabel('Cost function J(theta0)')
plt.xlabel('theta0')
plt.show()

plt.plot(thetaA1, costFunct)
plt.ylabel('Cost function J(theta)')
plt.xlabel('theta1')
plt.show()

plt.plot(costFunct[1:200])
plt.ylabel('Cost function J(theta)')
plt.xlabel('Number of Iteration')
plt.show()

print("Theta value : %d", theta)

# [-0.15393527,  0.23459561]

