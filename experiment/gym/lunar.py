from time import time, sleep

from gym.utils.env_checker import check_env
import gym
import pygame
from gym.utils.play import play, PlayPlot

# mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 1}
mapping = {(pygame.K_LEFT,): 3, (pygame.K_RIGHT,): 1, (pygame.K_UP, ):2}


# play(gym.make("CartPole-v0"), keys_to_action=mapping)
def callback(obs_t, obs_tp1, action, rew, done, info):
    return [rew, ]

plotter = PlayPlot(callback, 30 * 5, ["reward"])
# env = gym.make("Pong-v0")
env = gym.make("LunarLander-v2")
# env = gym.make("CartPole-v1")
# env.metadata["video.frames_per_second"] = 0.3
print(env.action_space)
play(env, callback=plotter.callback, keys_to_action=mapping, fps=12)

# env = gym.make("LunarLander-v2")
# print(check_env(env))
# # observation, info = env.reset(seed=42)
# # observation, info = env.reset()
# observation = env.reset()
# for _ in range(1000):
#     env.render()
#     action = env.action_space.sample()
#     # observation, reward, terminated, truncated, info = env.step(action)
#     observation, reward, done, info = env.step(action)
#
#     if done:
#         # observation, info = env.reset()
#         observation = env.reset()
#         # sleep(1000)
# env.close()

# Wrapper
# Transform actions before applying them to the base environment
# Transform observations that are returned by the base environment
# Transform rewards that are returned by the base environment
