from gym.envs.registration import register

register(
    id='deepracer-v0',
    entry_point='gym_deepracer.envs:DeepRacerEnv',
)

register(
    id='deepracerDiscrete-v0',
    entry_point='gym_deepracer.envs:DeepRacerEnvDiscrete',
)