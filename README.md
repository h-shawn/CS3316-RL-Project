# Reinforcement Learning Final Project

For Atari Games, I implemented DQN and Dueling DDQN. For Mujoco Robots, I implemented A3C and PPO.

## Structure

```shell
RL_final
├── atari
│   ├── atari_wrappers.py
│   ├── main.py
│   ├── memory.py
│   └── models.py
├── mujoco
│   ├── agent.py
│   ├── main.py
│   ├── memory.py
│   └── models.py
├── pic
│   ├── Ant.png
│   ├── Boxing.png
│   ├── Breakout.png
│   ├── HalfCheetah.png
│   ├── Hopper.png
│   ├── Humanoid.png
│   └── Pong.png
├── README.md
└── RL_final.pdf
```

## Usage

For value-based algorithm in atari

```shell
python3 main.py --env_name BreakoutNoFrameskip-v4 --is_dueling True --is_double True
```

For policy-based algorithm in mujoco

```shell
python3 main.py --env_name Hopper-v2 --method SAC
```

## Results

### Atari

<table>
    <tr>
        <td ><center><img src="pic/Boxing.png"></center></td>
        <td ><center><img src="pic/Breakout.png"></center></td>
      <td ><center><img src="pic/Pong.png"></center></td>
    </tr>
</table> 

### Mujoco

<table>
    <tr>
        <td ><center><img src="pic/Ant.png"></center></td>
        <td ><center><img src="pic/HalfCheetah.png"></center></td>
    </tr>
</table>  
<table>
    <tr>
      <td ><center><img src="pic/Hopper.png"></center></td>
      <td ><center><img src="pic/Humanoid.png"></center></td>
    </tr>
</table>
