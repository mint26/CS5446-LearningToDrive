# CS5446 - Learning to drive in a day

## Project Introduction

[Wayve](https://wayve.ai/) is a pioneering artificial intelligence software company for self-driving cars. Their unique end-to-end machine learning approach learns to drive in new places more efficiently than competing technology. There are two main algorithmic designs for the auto-driving system, which is the modular system and the end-to-end driving system. Modular systems, which are widely used in the real-world auto-driving systems, are structured as a pipeline of separate components linking sensory inputs to actuator outputs. Those components include localization and mapping, perception,
assessment, planning and decision making, vehicle control, and human-machine interface etc. However, modular systems have certain drawbacks; being prone to error propagation and over-complexity. End-to-end driving system, however, is a
promising next-generation auto-driving approach, which will generate ego-motion like steering wheel and pedals directly from sensory inputs. There are three main approaches for end-to-end driving systems: direct supervised deep learning, neuroevolution and the more recent deep reinforcement learning. We will focus on deep reinforcement learning in our project.

In this project, we will study Wayve’s [first application of deep reinforcement learning onboard an autonomous car](https://www.researchgate.net/publication/326144771_Learning_to_Drive_in_a_Day), which achieves the first real world run with Deep Q Networks (DQN) in a countryside road without traffic. For the implementation, the base code is forked from a [boilerplate git repository](https://github.com/r7vme/learning-to-drive-in-a-day). Further changes were made to cater to fix the dependencies and the experiment we wanted to conduct in this project.

## Project Setup

- The following prerequisite will be needed in order to have the application running in the machine:

  - Linux OS (Alternatively, can use [virtual box](https://www.virtualbox.org/) or [parallel desktop](https://www.parallels.com) if using Mac OS/Windows)
  - [Docker](https://docs.docker.com/engine/install/ubuntu/)
  - [Compiled Donkey Car Simulator](https://drive.google.com/open?id=1sK2luxKYV1cpaZLhVwfXrmGU3TRa5C3B)

- After having all the prerequisite fulfill, the application can be build by running the docker command. This command will install the libraries indicated in the requirement.txt.
  ```
  docker build -t learning-to-drive-in-a-day .
  ```

## Running the application

- Once the application is set up, you can run the application with the following command. (Note that by default it will train the agent using VAE.

  ```
  sudo ./run-in-docker.sh
  ```

- If you want to train the agent, without vae, you can do so by copy and paste the following command to `run-in-docker.sh` so it will point the entry point to `run-without-vae.py` instead)

```
docker run --net host \
    --rm \
    -ti \
    -e DISPLAY \
    -e DONKEY_SIM_HEADLESS=0 \
    -e DONKEY_SIM_PORT=9091 \
    -e DONKEY_SIM_PATH=./sim/donkey_sim.x86_64 \
    -v $HOME/.Xauthority:/root/.Xauthority \
    -v $HOME/sim:/sim \
    -v $(pwd):/code \
    -w /code \
    --device=/dev/dri:/dev/dri \
    learning-to-drive-in-a-day ./run-without-vae.py
```

- Note that the donkey car simulator need to be loaded first so that the connection between the application and simulation can be detected. Also, do check the error message. If it returns missing port number, do check the configured port number of the simulation application. Currently, its set to port 9091.

- Quick reference to run the application in simulator
  ![Step to run the application in simulator](https://user-images.githubusercontent.com/25121123/142888775-fbdb56d0-260c-485d-ba40-98b3a48d451b.gif)

## Project Structure

```

├── demo                        // folder containing the gifs to showcase the result for both DDPG only and DDPG + VAE
├── backup_model                // contains the train models used in this experiment
    ├── pretrained_model        // contains the pretrained model provided by the boilerplate repository
    ├── our_model               // contains our own model including ddpg only and ddpg + vae
├── result                      // contains the statistics recorded during training and testing phase
    ├── training result         // combined statistics recorded during the training phase
    ├── testing result          // recorded reward result during the testing phase using various model
├── vae                         // contains the code used to train the VAE
├── Dockerfile                  // docker configuration file used to build the docker image
├── .gitignore                  // indicates the files shouldn't be pushed to git
├── requirement.txt             // python config file on the dependencies
├── run-in-docker.sh            // bash script used to run the application
├── run-with-vae.py             // python file used to train or test the agent with VAE depends if the model exists
├── run-with-vae.py             // python file used to train or test the agent without VAE depends if the model exists
├── SetUnityLowResolution.sh    // used to set the unity to lower resolution if needed
├── README.md                   // this file
├── report.pdf                  // the report for the module final project


```

## Demonstration

Here's a demonstration of the donkey car experiment using DDPG + VAE.

### Agent in initial training phase

![Agent in initial training phase](https://user-images.githubusercontent.com/25121123/142881980-ab3ee95a-ff36-413b-a00c-68a78847c7b1.gif)

#### Agent in later training phase

![Agent in later training phase](https://user-images.githubusercontent.com/25121123/142882008-af4dc90e-7360-4c57-83ff-7282e800e436.gif)

## Notes

The codes can be further refactored for better performance and maintainability: </br>

- refactor the shell script to accept parameter to take in the two various approaches instead of manually change it in the script
- do experiment using other parameters to see whether it would work better like mentioned in the paper
- use other algorithms such as TD3 which might help in improving the accuracy of the navigation
