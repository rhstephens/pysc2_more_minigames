# PySC2 - More Minigames!

## Preface
This repository serves as a tutorial for making your own custom minigames as defined in [DeepMind](http://deepmind.com)'s StarCraft II Learning Environment (SC2LE). If you want to get familiar with some of the concepts, I recommend that you read the research paper that describes the SC2LE: [StarCraft II: A New Challenge for Reinforcement Learning](https://deepmind.com/documents/110/sc2le.pdf).



## Minigames

A **Minigame** is a controlled subset of StarCraft II's environment that allows agents to learn certain features of the game. Instead of having access to the entire map and action space, we can define Minigames for agents to learn one particular task (for example, training an SCV to harvest minerals and vespene gas). These Minigames are created like any other map in StarCraft, through the [StarCraft II Galaxy Editor](http://starcraft-2-galaxy-editor-tutorials.thehelper.net/tutorials.php).

An agent learns by taking observations from the environment (in the shape of feature layers) alongside its Curriculum Score (i.e. a reward that illustrates how well an agent doing right now). To create a successful Minigame, you have to design a challenge that *rewards* an agent when it's doing something right, and optionally *punishing* an agent when it's doing something wrong.

The most basic Minigame provided by the SC2LE is **MoveToBeacon**. An agent must learn how to use the basic move command by navigating to the glowing beacons.

`GIF HERE showing MoveToBeacon` 

As simple as it sounds, it takes the agent ~3000 attempts (episodes) to play this Minigame at the quality you are seeing above. That is why it is *crucial* to have a well designed Curriculum Scoring system in your Minigame, especially as the tasks get more complicated.

# Training an agent on a custom Minigame



## Setting up your environment


# Curriculum Training

Using an already trained agent to learn more advanced tasks. Using a 1 beacon agent to learn two beacons.



