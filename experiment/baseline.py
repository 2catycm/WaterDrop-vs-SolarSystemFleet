import gym

from stable_baselines3 import A2C


env = gym.make("CartPole-v1")
# env = gym.make("CartPole-v1", render_mode="human")


model = A2C("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=10_000)
model.learn(total_timesteps=10_00)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    # vec_env.render()
    vec_env.render(mode="human")
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()
# vec_env.close()