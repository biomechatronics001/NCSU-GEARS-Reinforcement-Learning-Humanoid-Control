

env = gym.make('CartPole-v1')             # Create an environment for the CartPole-v1 game using OpenAI Gym
states = env.observation_space.shape[0]   # Get the number of observations (state space dimensions) from the environment
actions = env.action_space.n              # Get the number of possible actions in the environment
observations = env.reset()                # Reset the environment to the initial state and obtain the initial observations
scoress = []                              # Initialize an empty list to store the scores for each episode
episodes = 30                             # Set the total number of episodes to run the agent

for episode in range(1, episodes+1):      # Loop over each episode from 1 to 30 (inclusive)
    state = env.reset()                   # Reset the environment to the initial state and get the initial state
    done = False                          # Initialize the 'done' flag to False, indicating the episode is not finished
    score = 0                             # Initialize the score for the current episode to 0

    while not done:                       # Loop until the episode is finished
        env.render()                      # Render the current state of the environment (useful for visualization)
        action = random.choice([0, 1])    # Randomly choose an action (0 or 1) from the action space
        n_state, reward, done, info = env.step(action)  # Take the chosen action and receive the next state, reward, done flag, and additional info
        score += reward                   # Accumulate the reward into the score for the current episode

    print('Episode:{} Score:{}'.format(episode, score))  # Print the episode number and the score obtained in that episode
    scoress.append(score)                 # Append the score of the current episode to the scoress list

# !apt-get install -y xvfb x11-utils          # Install Xvfb and x11-utils to enable virtual display capabilities on the system
# !pip install pyvirtualdisplay==0.2.*        # Install the pyvirtualdisplay library, version 0.2.*, to create and manage virtual displays

from pyvirtualdisplay import Display        # Import the Display class from the pyvirtualdisplay library
display = Display(visible=False, size=(1400, 900))  # Create a virtual display with dimensions 1400x900 and make it invisible
_ = display.start()                         # Start the virtual display

from gym.wrappers.monitoring.video_recorder import VideoRecorder  # Import VideoRecorder to record videos of the environment
before_training = "before_training.mp4"     # Set the filename for the video recording before training

video = VideoRecorder(env, before_training) # Initialize the VideoRecorder to record the environment and save it as before_training.mp4
env.reset()                                 # Reset the environment to its initial state and get the initial observation

for i in range(200):                        # Loop for 200 time steps
  env.render()                              # Render the current state of the environment (for visualization)
  video.capture_frame()                     # Capture the current frame and add it to the video recording
  observation, reward, done, info = env.step(env.action_space.sample())  # Take a random action, get the next state, reward, done flag, and info
  
video.close()                               # Close the video recorder to finalize the video file
env.close()                                 # Close the environment

from base64 import b64encode                # Import the b64encode function to encode binary data to base64 format
def render_mp4(videopath: str) -> str:      # Define a function to render an MP4 video as a base64-encoded string
  mp4 = open(videopath, 'rb').read()        # Read the binary data from the specified video file
  base64_encoded_mp4 = b64encode(mp4).decode()  # Encode the binary data to a base64 string and decode it to a normal string
  return f'<video width=400 controls><source src="data:video/mp4;' \
         f'base64,{base64_encoded_mp4}" type="video/mp4"></video>'  # Return an HTML string to display the video in an HTML page

from IPython.display import HTML            # Import HTML from IPython.display to display HTML content in Jupyter Notebooks
html = render_mp4(before_training)          # Generate the HTML string for the recorded video
HTML(html)                                  # Display the HTML string in the Jupyter Notebook to show the video

max(scores)                                  # Calculate and return the maximum score from the 'scores' list
plt.plot(range(episode), scoress)            # Plot the scores of each episode. 'range(episode)' provides the x-axis values (episode numbers), and 'scoress' provides the y-axis values (scores)
plt.rcParams["figure.figsize"] = (2,2)       # Set the size of the figure to 2x2 inches using the 'figure.figsize' parameter in matplotlib's rcParams
plt.show()                                   # Display the plot on the screen

##################
## Next example ##
##################

env2 = gym.make('CartPole-v1')                        # Create a new environment instance for the CartPole-v1 game using OpenAI Gym
actions = env2.action_space.n                         # Get the number of possible actions in the environment (e.g., 2 for left and right)
state_shape = env2.observation_space.shape[0]         # Get the number of observations (state space dimensions) from the environment

def Building_Model(states, actions):                  # Define a function to build a neural network model
    model = Sequential()                              # Initialize a sequential model (a linear stack of layers)
    model.add(Flatten(input_shape=(1,states)))        # Add a flatten layer to reshape the input to a 1D array
    model.add(Dense(24, activation='relu'))           # Add a dense (fully connected) layer with 24 units and ReLU activation
    model.add(Dense(24, activation='relu'))           # Add another dense layer with 24 units and ReLU activation
    model.add(Dense(actions, activation='linear'))    # Add a dense layer with the number of actions and linear activation (for Q-values)
    return model                                      # Return the constructed model

model = Building_Model(state_shape, actions)          # Build the model using the defined function with state and action dimensions
model.summary()                                       # Print a summary of the model architecture

def Building_Agent(model, actions):                   # Define a function to build the reinforcement learning agent
    policy = BoltzmannQPolicy()                       # Define a Boltzmann Q-policy for action selection
    memory = SequentialMemory(limit=50000, window_length=1)  # Initialize memory with a limit of 50,000 steps and window length of 1
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                  nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)  # Initialize a DQN agent with the model, memory, policy, number of actions, warm-up steps, and target model update frequency
    return dqn                                        # Return the constructed DQN agent

dqn2 = Building_Agent(model, actions)                 # Build the DQN agent using the defined function with the model and action dimensions
dqn2.compile(tf.keras.optimizers.legacy.Adam(learning_rate=1e-3), metrics=['mae'])  # Compile the DQN agent with Adam optimizer and mean absolute error metric

display2 = Display(visible=False, size=(1400, 900))   # Create a virtual display with dimensions 1400x900 and make it invisible
display2.start()                                      # Start the virtual display

try:                                                  # Try block to ensure the display stops even if there is an error during training
    dqn2.fit(env2, nb_steps=5000, visualize=False, verbose=2)  # Train the DQN agent in the environment for 5000 steps without visualization and with verbose level 2
finally:
    display2.stop()                                   # Ensure the display stops even if an error occurs

after_training = "after_training.mp4"                 # Set the filename for the video recording after training
after_video = VideoRecorder(env2, after_training)     # Initialize the VideoRecorder to record the environment and save it as after_training.mp4
observation = env2.reset()                            # Reset the environment to its initial state and get the initial observation
done = False                                          # Initialize the 'done' flag to False, indicating the episode is not finished

while not done:                                       # Loop until the episode is finished
    after_video.capture_frame()                       # Capture the current frame and add it to the video recording
    action = dqn2.forward(observation)                # Use the trained DQN agent to select an action based on the current observation
    observation, reward, done, info = env2.step(action)  # Take the chosen action, get the next state, reward, done flag, and additional info

after_video.close()                                   # Close the video recorder to finalize the video file
env2.close()                                          # Close the environment

# Display the video
html = render_mp4(after_training)                     # Generate the HTML string for the recorded video
HTML(html)                                            # Display the HTML string in the Jupyter Notebook to show the video

scores = dqn2.test(env2, nb_episodes=50, visualize=False)  # Test the trained DQN agent for 50 episodes without visualization
print(np.mean(scores.history['episode_reward']))      # Print the mean reward per episode over the 50 test episodes
max(scores.history['episode_reward'])                 # Get the maximum reward achieved in any single episode
plt.plot(range(50), scores.history['episode_reward']) # Plot the reward obtained in each of the 50 test episodes
plt.rcParams["figure.figsize"] = (2,2)                # Set the size of the figure to 2x2 inches using the 'figure.figsize' parameter in matplotlib's rcParams
plt.show()                                            # Display the plot on the screen
