# graph_nets_ids576
Based on relational inductive bias, deep learning and graph networks - https://arxiv.org/abs/1806.01261

Graph networks are hybrid of structured graphs and deep learning. Input to the network is a graph and output is a learned graph with updated attributes of nodes, edges and global entity.

Demonstrations:

1. graph_nets_basics: go over the creation and manipulation of graph-structured data with GraphsTuple class and utils_tf and utils_np utilities provided by graph_nets library

2. shortest_path_demonstration: this demo creates random graphs, and trains a graph network to label the nodes and edges on the shortest path between any two nodes. The model refines its prediction of the shortest path over a sequence of message-passing steps (as depicted by each step's plot).

3. sort_demonstration: this demo creates lists of random numbers, and trains a graph network to sort the list. After a sequence of message-passing steps, the model makes an accurate prediction of which elements (columns in the figure) come next after each other (rows).

Key insights:

For upto 15 nodes per network, the model could achieve below 10% training and generalization loss in 5000 training iterations in both the shortest path and sort demos. The loss is calculated after each training step. The figures on loss vs. iterations and the visualization of the results can be observed at the end of the documents - shortest_path_demonstration.ipynb and sort_demonstration.ipynb

The model I explored includes three components:
 - An "Encoder" graph network, which independently encodes the edge, node, and
   global attributes (does not compute relations etc.).
 - A "Core" graph network, which performs N rounds of processing (message-passing)
   steps. The input to the Core is the concatenation of the Encoder's output
   and the previous output of the Core (labeled "Hidden(t)" below, where "t" is
   the processing step).
 - A "Decoder" graph network, which independently decodes the edge, node, and
   global attributes (does not compute relations etc.), on each
   message-passing step.
 
 ![alt text](https://github.com/priyeshshukla/graph_nets_ids576/blob/master/gn.PNG?raw=true)

 The model is trained by supervised learning. Input graphs are procedurally
 generated, and output graphs have the same structure with the nodes and edges
 of the shortest path labeled (using 2-element 1-hot vectors).

 The training loss is computed on the output of each processing step. The
 reason for this is to encourage the model to try to solve the problem in as
 few steps as possible and it also helps make the output of intermediate steps
 more interpretable.

 Evaluate of how well the models generalize to graphs is also done which are up to
 twice as large as those on which it was trained. The loss is computed only
 on the final processing step.

 Variables with the suffix _tr are training parameters, and variables with the
 suffix _ge are test/generalization parameters.

 After around 3000-5000 training iterations the model reaches near-perfect
 performance on graphs with between 8-16 nodes.
