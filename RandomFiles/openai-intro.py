#Import OpenAi
import gym

#Create environemt
env = gym.make(“FrozenLake-v0”)

#Resetting the environemt gives us our first state
state = env.reset()

for step in range(10):
    #Choose a random action
    action = env.action_space.sample()

    #Get observations
    new_state, reward, done, info = env.step(action)
    
    #Print things out to see what was returned
    print("state, new_state, reward, done, info = ", state, new_state, reward, done, info)

    #Set state as the new_state
    state = new_state

    #If our agent crashed or finished then break
    if done:
        break
    
