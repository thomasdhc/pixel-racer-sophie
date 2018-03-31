# Pixel Racer with Reinforcement Learning
#### Sohpie's Objective
Our little green pixel racer wants to reach the end of the track. Being a competitive little racer, she wants to reach the goal with the least amount of moves! She's going to practice and learn.

## Q-table Learning
With Q-table learning Sophie keeps track of how valuable all possible moves are from every pixel location.

Training Day 1             |  Training Day 55          |  Training Day 500
:-------------------------:|:-------------------------:|:-------------------------:
![](gifs/q_table_actions_0.gif)  |  ![](gifs/q_table_actions_1.gif) | ![](gifs/q_table_actions_2.gif)


## Q-learning using a Neural Network
Sohpie has big dreams! Though she trains on a small track now, she wants compete in the bigger stages and keeping track of every move on every pixel would be very intensive on her memory. Let's resolve this issue by training Sophie with a neural network.

Training Day 1             |  Training Day 55          |  Training Day 500
:-------------------------:|:-------------------------:|:-------------------------:
![](gifs/q_nn_actions_0.gif)  |  ![](gifs/q_nn_actions_1.gif) | ![](gifs/q_nn_actions_2.gif)

With the same number training steps the results using a neural network is not as impressive as the q-table, but she is able to reach the end effectively to a degree. 

Loss Graph                  |
:--------------------------:|
![](graphs/20180316_q_nn_loss.png)|

For this implementation the input to the neural network is a flattened array containing only Sophie's location.

In the future, Sophie would want to be aware of all her surrondings such as obstacles, enemies, etc. Let us train Sophie while preserving all this information.

## Q-learning using a Neural Network and a Flattened Array

Training Day 1             |  Training Day 500          |  Training Day 1000
:-------------------------:|:-------------------------:|:-------------------------:
![](gifs/q_array_0.gif)  |  ![](gifs/q_array_1.gif) | ![](gifs/q_array_2.gif)

The additional environment information prevents Sohpie from reaching the end. New complexities arise with the introduction of the new variables. We decreased Sohpie's learning rate to prevent gradient descent from diverging, but the reduction stops Sohpie from effectively learning which move she must make.

## Experience Buffer and a Target Neural Network

We want Sohpie to learn from a variety of pixel locations and moves. Currently most of Sohpie's training takes place near the starting point. By mixing up her experience, we are able to give her a more balanced training at each step. This will be done using a experience buffer.

Secondly, we will create a second network that provides the target value at each state. Arthur Juliani describes it best in his [deep reinforcement learning medium post](https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-4-deep-q-networks-and-beyond-8438a3e2b8df): 
> The issue with using one network for training is that at every step the Q-network’s values shift. Using a constantly shifting set of values to adjust our network values, the value estimation can spiral out of control. Network becomes destabilized by falling into feedback loops between the target and estimated Q-values.

## The Final Implementation

First, we give Sohpie 10,000 steps to take randomly on the track. We store her experiences in a buffer. Throughout her training we give Sohpie random samples. Using the target network and her main network, we train her to reach the end. As the training process proceeds we replace part of the experience buffer with more recent experiences and move weights of the target network slowly towards the weights of the main network.

Volia!

Training Day 10,000       |
:------------------------:|
![](gifs/buffer_q_nn.gif) |

Loss graph                 |
:-------------------------:|
![](graphs/20180327_buffer_q_nn_loss.png) |

Now that Sohpie is aware of herself and her environment, we must watch out for her world domination!

## Getting Started
#### Installation
You need Python3 to run Pixel Racer Sophie. 

Dependency installation:
```
python setup.py install
```
To train the models for yourself:
```
python -m model.q_nn run
```
On completion you will be presented with a loss graph and the training actions will be stored in model_results.
You can view your racer's actions:
```
python -m environment.track_world run <name_of_action_file>
```
This will generate animation plot of your racers actions.
Feel free to build your own tracks by modifying the track world file and train your racer on the new track.

## Future

Sophie's training continues ...

## Thank you and Further Readings

I learned a great amount from the reinforcement learning series by Arthur Juliani.

https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0

For a deeper technical read on reinforcment learning I would recommend this series.

He also links in his series to another great article about deep reinforcement learning, which I found very helpful.

http://neuro.cs.ut.ee/demystifying-deep-reinforcement-learning/