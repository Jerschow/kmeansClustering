import input_validation as iv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import xlim,ylim


class Node:
    def __init__(self,x,y,name):
        self.x = x
        self.y = y
        self.name = name
    
    def get_dist(self,other):
        finddot = np.array([self.x - other.x,self.y - other.y])
        return np.dot(finddot,finddot)


def get_data_hypers():
    ptlist = [Node(1,6,"A"),Node(1002,20,"B"),Node(498,651,"C"),Node(6,10,"D"),Node(510,622,"E"),Node(503,632,"F"),Node(4,9,"G"),Node(1010,25,"H"),Node(1006,30,"I"),\
        Node(502,680,"J")]
    points = np.zeros((len(ptlist),3))
    for i in np.arange(len(ptlist)):
        node = ptlist[i]
        rowi = [node.x,node.y]
        rowi.append(None)
        points[i] = np.array(rowi)
    centers = [None] * 3
    if iv.validate("Use A,B,C as centers or predetermined vals (y for abc)?") == 'y':
        for i in np.arange(len(centers)):
            centers[i] = ptlist[i]
    else:
        centers = [Node(500,10,"c1"),Node(200,700,"c2"),Node(800,200,"c2")]
    return ptlist,points,centers

def classify(tolerance,eta,points,ptlist,centers):
    diff = np.inf
    i = 0
    while diff > tolerance and i < 10e9:
        diff = - np.inf
        # this next loop gets the corresponding center for each point
        for j in np.arange(len(ptlist)):
            mindist = np.inf
            for k in np.arange(len(centers)):
                jkdist = ptlist[j].get_dist(centers[k])
                print("distance from " + centers[k].name + " to " + ptlist[j].name + ": " + str(jkdist))
                if jkdist < mindist:
                    mindist = jkdist
                    points[j,2] = k
            print(ptlist[j].name + " classified as " + str(points[j,2]))
        newcenters = [None] * 3
        print()
        # this next loop gets the gradient for each center and gets the new center
        for j in np.arange(len(centers)):
            grad = np.array([0,0])
            for k in np.arange(len(points)):
                if points[k,2] == j:
                    kgrad = np.array([ptlist[k].x - centers[j].x,ptlist[k].y - centers[j].y])
                    print("pseudograd of " + ptlist[k].name + " for cluster " + centers[j].name + ":\n" + str(kgrad))
                    grad = np.add(grad,kgrad)
            print("pseudograd of for cluster " + centers[j].name + ":\n" + str(grad))
            newx = centers[j].x + eta * grad[0]
            newy = centers[j].y + eta * grad[1]
            newcenters[j] = Node(newx,newy,centers[j].name)
        print("New centers:")
        printable = []
        for j in np.arange(len(newcenters)):
            printable.append("x: " + str(newcenters[j].x) + ",y: " + str(newcenters[j].y))
        print(printable)
        print()
        for j in np.arange(len(centers)):
            diff = max(diff,centers[j].get_dist(newcenters[j]))
        centers = newcenters
        i += 1
    return points,centers


ptlist,points,centers = get_data_hypers()
points,centers = classify(.01,.1,points,ptlist,centers)
centersnp = np.zeros((3,2))
for i in np.arange(len(centers)):
    centersnp[i] = np.array([centers[i].x,centers[i].y])
plt.scatter(points[:,0],points[:,1],c=points[:,2])
plt.scatter(x=centersnp[:,0],y=centersnp[:,1],c=np.arange(3),marker="X",s=200)
xlim(min(points[:,0]) - 10,max(points[:,0]) + 10)
ylim(min(points[:,1]) - 10,max(points[:,1]) + 10)
plt.show()