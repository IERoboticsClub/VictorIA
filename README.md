![hacktoberfest](https://img.shields.io/badge/Hacktoberfest-2023-blueviolet?style=for-the-badge&logo=appveyor)

# VictorIA

VictorIA is an intelligent robotic player from the IE Robotics & AI Club that is powered by AI to play a series of different physical games through the use of the dobot magician robotic arms. 


In this case we will program the robot to play **Connect-4**.

![connect4](https://github.com/IERoboticsClub/VictorIA/assets/63413550/08fbe475-5cd8-4606-9736-49cb82d6e6a4)

## Development Setup

1. Create and activate a virtual environment
```bash
python3 -m venv venv && . venv/bin/activate
```

2. Install dependencies
```bash
pip install -r requirements.txt
```


## Contributing

Follow the steps in `CONTRIBUTING.md` to contribute to this project.


## How can you contribute?
There are two ways you can contribute in the project:

1. **Computer Vision**:  we have 2 notebooks in the tests folder that perform circle detection and color detection respectively. You can try to improve or create your own file to solve the computer vision part. We want the output of this part to be a matrix representing the state of the board.
![computervision](https://github.com/IERoboticsClub/VictorIA/assets/63413550/0847cd5e-caad-4392-a4a7-b3a9a1277bf1)

2. **The resolution algorithm**: we did not start this part, so you can come with any idea: MiniMax, MCTS.... We are thinking of creating different levels of difficulty, so maybe you can use the same method with different heuristics depending on the level. We want that, given the representative matrix of the board, the algorithm you program, gives as output the column where the new chips go. 
![resolution_algorithm](https://github.com/IERoboticsClub/VictorIA/assets/63413550/9be7cc38-97b2-45df-9d27-1140ad8dc3b1)



