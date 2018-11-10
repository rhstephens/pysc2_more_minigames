# PySC2 - More Minigames!

This repository serves as a tutorial for making your own custom minigames as defined in [DeepMind](http://deepmind.com)'s StarCraft II Learning Environment (SC2LE). If you want to get familiar with some of the concepts, I recommend that you read the research paper that describes the SC2LE: [StarCraft II: A New Challenge for Reinforcement Learning](https://deepmind.com/documents/110/sc2le.pdf).

## Minigames

A **Minigame** is a controlled subset of StarCraft II's environment that allows agents to learn certain features of the game. Instead of having access to the entire map and action space, we can define Minigames for agents to learn one particular task (for example, training an SCV to harvest minerals and vespene gas). These Minigames are created like any other map in StarCraft, through the [StarCraft II Galaxy Editor](http://starcraft-2-galaxy-editor-tutorials.thehelper.net/tutorials.php).

An agent learns by taking observations from the environment (in the shape of feature layers) alongside its `Curriculum Score` (i.e. a reward that illustrates how well an agent doing right now). To create a successful Minigame, you have to design a challenge that *rewards* an agent when it's doing something right, and optionally *punishing* an agent when it's doing something wrong.

The most basic Minigame provided by the SC2LE is `MoveToBeacon`. In this Minigame, an agent must learn how to use the basic move command by navigating towards the glowing beacons.

![](https://github.com/codetroopa/pysc2_more_minigames/raw/master/screenshots/MoveToBeacon/one_beacon_example.gif "MoveToBeacon")

As simple as it sounds, it takes the agent ~3000 attempts (episodes) to play this Minigame at the quality you are seeing above. That is why it is *crucial* to have a well designed `Curriculum Scoring` system in your Minigame, especially as the tasks and environment get more complicated.

# Training an agent against a custom Minigame

Here, I will guide you through making a custom Minigame that an agent can learn to play. We will go through the creation of **MoveTwoBeacons** and then training an agent to play it correctly. Before we start working in the StarCraft II Galaxy Editor and design our new Minigame, we need to set a few things up.

## Setting up your environment

To get started, you will need:
- StarCraft II (at least v3.16.1)
- [pysc2](https://github.com/deepmind/pysc2) (**v1.2**) with the [StarCraft II Minigames](https://github.com/deepmind/pysc2#get-the-maps)
- Python 3 (tested on 3.6.6)
- Tensorflow / Tensorflow-gpu (tested on 1.11.0)

I have decided to use simonmeister's [pysc2-rl-agents](https://github.com/simonmeister/pysc2-rl-agents) to train my agents. You will need to clone that repo into your working directory for later use.

## Creating a Minigame

### The StarCraft II Galaxy Editor

The Galaxy Editor should come already installed when you downloaded the game so go ahead and open it up from the Battle.net launcher. We are going to use `MoveToBeacon.SC2Map` as a base, so copy the file and rename it to `MoveTwoBeacons.SC2Map`. These Minigames should be located in `<install_path>/StarCraft II/Maps/minigames/` after following the directions above. Open up the new SC2Map in the Editor and you should be welcomed with this:

![](https://github.com/codetroopa/pysc2_more_minigames/raw/master/screenshots/MoveTwoBeacons/editor.png "Editor")

### Terrain Module

What you are seeing is the `Terrain Module`, which allows you to place Units, Regions, Doodads, Cameras etc. as well as modifying the Terrain itself. For our purposes, we only need to worry about `Regions`.

1. Click this to bring up the `Region` view.
2. Here we can add new regions or modify existing ones. Since we need two beacons for this Minigame, click the Circle region selector and drag it onto the screen. This will create another region which you can name `Beacon Area 2`. Copy the values from `Beacon Area` to make sure they match.

I've noticed that the region size of `Beacon Area` is actually smaller than the visual size of the beacon when placed, which has an effect on the agent's outcome. I recommend changing the `Beacon Area` size from 2.50 -> 3.0

### Trigger Module

That's all we need from the Terrain Module. Next, we want to work on this Minigame's `Triggers`. Triggers allow us to modify the game state whenever a certain event happens. This event can be almost anything that occurs in StarCraft II - from an amount of time elapsing to a certain Unit using an ability. The general flow of a Trigger is as follows:

 1. **Events:** Some event in the game has happened, activating an instance of this trigger.
 2. **Conditions:** Before performing any actions, ensure that *all* of the listed conditions are satisfied. If any condition returns false, no actions are performed.
 3. **Actions:** A list of actions carried out in sequential order.

There are plenty of user-made tutorials on triggers and if you get lost going forward, check out [SC2Mapster Tutorials](https://sc2mapster.gamepedia.com/Tutorials).

To start working with triggers, open up the `Trigger Module` shown at #3 above. You will see this:

![](https://github.com/codetroopa/pysc2_more_minigames/raw/master/screenshots/MoveTwoBeacons/triggers_old.png "Trigger Module")

The left-side panel displays a list of defined triggers and variables used to manipulate the game state. Most of the work is done for us, but we will have to make a few tweaks.

Before diving right in, I should let you know that you can test your minigame at any time by pressing the green StarCraft II icon (Ctrl + F9) from the Terrain Module screen. It's useful to frequently test your minigame as you add more and more functionality to it.

To start, we need to define a few variables (right click the panel and select New->New Variable):
- Copy `Beacon` so that we have a reference to our second beacon unit.
- Create two **Boolean** variables named "Beacon 1 Reached" and "Beacon 2 Reached" respectively. We will need these to determine when it's time to reset our beacon locations.

There are a few times when we want to reset the location of our beacons, so it makes sense to create a *custom Action*. A custom Action allows us to define our own sequence of Actions with optional parameters and a return type. Once defined, we can call that Action whenever we need it.

To define a new Action, right click the left-side panel and select New->New Action Definition. Here's what we want it to do:

![](https://github.com/codetroopa/pysc2_more_minigames/raw/master/screenshots/MoveTwoBeacons/triggers_reset_beacons_action.png "Reset Beacons Action")

1. Destroy the current references to `Beacon` and `Beacon 2`. Then, create new beacons at a random point in `Beacon Spawn Region`. Make sure our variables reference the newly created units.
2. In a while loop, move the beacon to a new random point in the region until our conditions are satisfied. We want to ensure that
    -  The new beacon isn't trivially close to our Marine.
    -  The distance between the two beacons is non-trivial.
    -  Once those are satisfied, we move the actual `Beacon Area` region to the new location.
3. Same as above, but for `Beacon 2` and `Beacon Area 2`.
4. Reset our variables used to track when beacons have been reached.

We have ourselves a handy custom Action, but haven't used it yet. Next, let's modify our Init trigger. This trigger is called once at the beginning of the game and sets up some required properties.

![](https://github.com/codetroopa/pysc2_more_minigames/raw/master/screenshots/MoveTwoBeacons/triggers_init.png "Init Trigger")

1. All we need to do here is remove the old set of actions that initialized the beacon and replace it with our newly created `Reset Beacons` custom Action. The rest of the trigger can be left alone as we need it to set up things like the Camera, Curriculum Score, and other triggers.

Now we need to modify our `Curriculum Scoring` trigger. What we want is a trigger that checks if our Marine has entered one of the two beacons. If so, we add 1 to our score and disable that beacon (temporarily).

Copy the existing `Score Updates and Victory` trigger as a base to create the following two triggers:

![](https://github.com/codetroopa/pysc2_more_minigames/raw/master/screenshots/MoveTwoBeacons/triggers_beacon_score1.png "Score 1")

![](https://github.com/codetroopa/pysc2_more_minigames/raw/master/screenshots/MoveTwoBeacons/triggers_beacon_score2.png "Score 2")
   
Be sure to include the `Beacon Reached` conditions and to update the variable after entering the region. Otherwise, you could be double counting scores.

Now, we need a trigger to reset the beacons once both have been reached by the Player:

![](https://github.com/codetroopa/pysc2_more_minigames/raw/master/screenshots/MoveTwoBeacons/triggers_check_for_reset.png "Check For Reset")

This one is easy. All we need is a Timer event that fires every so often, checking if both `Beacon 1 Reached` and `Beacon 2 Reached` are true. If so, we call our custom `Reset Beacons` Action made earlier.

When we eventually train an Agent to play our minigame, there has to be a quick way for the Agent to reset the minigame without performing a full restart. The Agent does this by typing "reset" into the game chat, expecting the game to handle the rest. When designing our minigame, we have to implement this reset functionality in a Trigger. 

There already exists a trigger named `Reset Map` that we have to modify for our purposes. You can leave everything as is except for the following:

![](https://github.com/codetroopa/pysc2_more_minigames/raw/master/screenshots/MoveTwoBeacons/triggers_reset_map.png "Reset Map")

1. This temporarily disables scoring while resetting the map state.
2. Reset the beacons to a valid position.
3. Reactivate scoring Triggers.

*And that's it!*

I'd recommend poking around the other Triggers/Variables so you have a full understanding of the minigame.

## Training the Agent

Now that we have a minigame, it's time to train an Agent to play it.

# Minigame Results

## MoveTwoBeacons

The first Minigame I made was a simple modification of `MoveToBeacon`.

![](https://github.com/codetroopa/pysc2_more_minigames/raw/master/screenshots/MoveTwoBeacons/two_beacons_example.gif "MoveTwoBeacons")

Instead of having a single beacon spawn each time a marine enters one, there will be two beacons that the marine must navigate to before the next two spawn. The idea is to teach an agent not only how to navigate to a location, but how to choose an optimal route to get there.

After training the agent from scratch for ~5000 episodes, you can tell that the agents performs well, but still makes some mistakes on finding the shortest route.

### Episode Score

![](https://github.com/codetroopa/pysc2_more_minigames/raw/master/screenshots/MoveTwoBeacons/two_beacons_episode_score.png "MoveTwoBeacons episode_score")

## Score Comparisons

Mean scores were taken over 100 episodes for the Agent to see how it compares against a real Human player (me) over 10 episodes.

| Minigame | Mean Episode Score (Trained Agent) |  Mean Episode Score (Human) |
| --- | --- | --- |
| MoveToBeacon | **26.76** | 26.4 |
| MoveTwoBeacons | **26.74** | 29.7 |

# Curriculum Training

Cite Cirriculum Training paper here.

Using an already trained agent to learn more advanced tasks. Using a 1 beacon agent to learn two beacons.

Watch what happens when we test our `MoveToBeacon` agent against our new `MoveTwoBeacons` map:



