# Distributed Algorithm Optimization
Perceptron algorithm is widely used in the natural language processing and the
machine learning. It intrinsically requires that the whole dataset is kept in one machine.
In the distributed systems, however, to improve the speed of the calculation, an important
strategy is dividing the whole dataset into several partitions across the clusters and
performing parallel calculations.  Here, I proposed several models to solve this
contradiction. The results showed that my models decrease the number of messages
among the clusters and improve the performance.  I implemented Perceptron algorithm
in the distributed systems with Apache Spark on Amazon Elastic Compute Cloud (EC2).
 With this design, I have achieved the machine learning with security, high scalability,
high speed, and fault tolerance.

# Methods
<p></p>
![Alt text](https://github.com/Charley-Wang/Distributed-Algorithm-Optimization/blob/master/results/Model0.jpg?raw=true "Main Interface")
<p></p>
![Alt text](https://github.com/Charley-Wang/Distributed-Algorithm-Optimization/blob/master/results/Model1.jpg?raw=true "Main Interface")
<p></p>
![Alt text](https://github.com/Charley-Wang/Distributed-Algorithm-Optimization/blob/master/results/Model2.jpg?raw=true "Main Interface")
<p></p>
![Alt text](https://github.com/Charley-Wang/Distributed-Algorithm-Optimization/blob/master/results/Model3-6.jpg?raw=true "Main Interface")

# Results
### The Average Errors of the Last 50 Iterations Well Represent the Performance
<p></p>
![Alt text](https://github.com/Charley-Wang/Distributed-Algorithm-Optimization/blob/master/results/Figure1.jpg?raw=true "Main Interface")
<p></p>
Figure 1. The moving average errors (%) of training, validation, and testing datasets. The
window size for the average is 10. The learning is based on the Small Dataset with the
serial Model.

### Model 3 Has the Best Performance
<p></p>
![Alt text](https://github.com/Charley-Wang/Distributed-Algorithm-Optimization/blob/master/results/Table1.jpg?raw=true "Main Interface")
<p></p>
Table 1. The last 50 average errors (ERRlast50, %) for Model 0 to Model 5 on the Small Dataset.
<p></p>
![Alt text](https://github.com/Charley-Wang/Distributed-Algorithm-Optimization/blob/master/results/Table2.jpg?raw=true "Main Interface")
<p></p>
Table 2. The last 50 average errors (ERRlast50, %) for Model 0 to Model 6 on the Big Dataset.

### In-memory Cluster Computing Fasts the Learning 
Spark provides Cache function on
RDDs that caches dataset in the memory of each cluster. It fasts the machine learning
because of avoiding reading data from the disk for each iteration. To test this function, I
performed the learnings with Model 1 on the Big Dataset with or without applying the
Cache function. The time for the learning without cached data is 820 seconds. With
cached data, the time for the learning is 538. It shows that in-memory cluster computation
fasts the machine learning.
