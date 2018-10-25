# PySC2 - More Minigames!

This repository serves as a tutorial for making your own custom minigames as defined in [DeepMind](http://deepmind.com)'s StarCraft II Learning Environment (SC2LE). If you want to get familiar with some of the concepts, I recommend that you read the research paper that describes the SC2LE: [StarCraft II: A New Challenge for Reinforcement Learning](https://deepmind.com/documents/110/sc2le.pdf).

## Minigames

A **Minigame** is a controlled subset of StarCraft II's environment that allows agents to learn certain features of the game. Instead of having access to the entire map and action space, we can define Minigames for agents to learn one particular task (for example, training an SCV to harvest minerals and vespene gas). These Minigames are created like any other map in StarCraft, through the [StarCraft II Galaxy Editor](http://starcraft-2-galaxy-editor-tutorials.thehelper.net/tutorials.php).

An agent learns by taking observations from the environment (in the shape of feature layers) alongside its `Curriculum Score` (i.e. a reward that illustrates how well an agent doing right now). To create a successful Minigame, you have to design a challenge that *rewards* an agent when it's doing something right, and optionally *punishing* an agent when it's doing something wrong.

The most basic Minigame provided by the SC2LE is `MoveToBeacon` An agent must learn how to use the basic move command by navigating towards the glowing beacons.

```
GIF HERE showing MoveToBeacon
```

As simple as it sounds, it takes the agent ~3000 attempts (episodes) to play this Minigame at the quality you are seeing above. That is why it is *crucial* to have a well designed `Curriculum Scoring` system in your Minigame, especially as the tasks and environment get more complicated.

# Minigame Results

## MoveTwoBeacons

The first Minigame I had made was a simple modification of `MoveToBeacon`.

```
GIF HERE showing MoveTwoBeacons
```
Instead of having a single beacon spawn each time a marine enters one, we will have two beacons that the marine must navigate to before the next two spawn. The idea is to teach an agent not only how to navigate to a location, but how to choose an optimal route to get there. The rest of this tutorial will guide you through the creation of **MoveTwoBeacons** and training an agent to play it correctly.

# Training an agent against a custom Minigame

Here, I will guide you through making a custom Minigame that an agent can learn to play. Before we start working in the StarCraft II Galaxy Editor and design our new Minigame, we need to set a few things up.

## Setting up your environment

To get started, you will need:
- StarCraft II (at least v3.16.1)
- [pysc2](https://github.com/deepmind/pysc2) (**v1.2**)
- [StarCraft II Minigames](https://github.com/deepmind/pysc2#get-the-maps)
- Python 3 (tested on 3.6.6)
- Tensorflow / Tensorflow-gpu (tested on 1.11.0)

I have decided to use simonmeister's [pysc2-rl-agents](https://github.com/simonmeister/pysc2-rl-agents) to train my agents. You will need to clone that repo into your working directory for later use.

## The StarCraft II Galaxy Editor

The Galaxy Editor should come already installed when you downloaded the game so go ahead and open it up from the Battle.net launcher. We are going to use `MoveToBeacon.SC2Map` as a base, so copy the file and rename it to `MoveTwoBeacons.SC2Map`. These Minigames should be located in `<install_path>/StarCraft II/Maps/minigames/` after following the directions above. Open up the new SC2Map in the Editor and you should be welcomed with this:

![](https://github.com/codetroopa/pysc2_more_minigames/raw/master/screenshots/editor.png "Editor")


# Curriculum Training

Using an already trained agent to learn more advanced tasks. Using a 1 beacon agent to learn two beacons.

Watch what happens when we test our `MoveToBeacon` agent against our new `MoveTwoBeacons` map:



