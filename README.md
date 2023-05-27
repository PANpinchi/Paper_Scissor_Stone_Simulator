# Rock Paper Scissors Simulator
This is a rock paper scissors simulator that simulates the movement and interaction of rocks, papers, and scissors on a map until only one type of element remains.

## Features
- Create a 1000 x 1000 map with different elements represented by numbers:
  - 0 represents rock
  - 2 represents scissors
  - 5 represents paper
- Randomly distribute 10 rocks, scissors, and papers on the map
- Movement of elements:
   - Rocks move towards the nearest scissors
   - Scissors move towards the nearest paper
   - Paper moves towards the nearest rock
- Interaction rules:
   - If a rock moves next to a pair of scissors, the scissors turn into a rock
   - If scissors move next to a piece of paper, the paper turns into scissors
   - If paper moves next to a rock, the rock turns into paper
- Repeat the above movement and interaction until only one type of element remains

## Usage
1. Install Python environment (Python 3.x is recommended)
2. Install the required library (NumPy):
``` XML
  pip install numpy
```
3. Download and run the rock paper scissors simulator program:
``` XML
  python main.py
```
4. The program will generate a map and initialize the distribution of elements, and start simulating the movement and interaction process.
5. When only one type of element (rock, scissors, or paper) remains, the simulator will stop and display the final result.

## Demo


