# ♠️ Blackjack Reinforcement Learning Agent

A Python implementation of a Reinforcement Learning agent that learns to play a simplified version of Blackjack using **Monte Carlo First-Visit methods**.

## Overview

This project builds a custom Blackjack environment and an RL agent from scratch. The agent treats the game as a **Markov Decision Process (MDP)** and learns the optimal strategy (Policy) through trial and error over 5,000,000 episodes.

For simplicity and faster convergence, this model uses a **"One Color"** deck (ignoring suits), focusing purely on card values and probability.

### Key Concepts
* **Monte Carlo Method:** The agent learns from complete episodes (games played to the end) rather than bootstrapping.
* **First-Visit MC:** The Q-value is updated using the return from the *first* time a state was visited in an episode.
* **Epsilon-Greedy Policy:** The agent explores random moves 10% of the time to ensure it discovers new strategies, while exploiting its best knowledge 90% of the time.

## The Environment

The environment is a simplified simulation of Blackjack:

| Component | Description |
| :--- | :--- |
| **State Space** | `(Player Sum, Dealer Showing Card, Usable Ace)` |
| **Actions** | `0` (Stick), `1` (Hit) |
| **Rewards** | `+1` (Win), `-1` (Loss/Bust), `0` (Draw) |
| **Simplification** | Infinite deck, single suit (uniform probability for 2-9, 4x for 10s). |

## Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/blackjack-rl.git](https://github.com/yourusername/blackjack-rl.git)
    cd blackjack-rl
    ```

2.  **Install dependencies:**
    ```bash
    pip install numpy matplotlib
    ```

3.  **Run the Training:**
    ```bash
    python blackjack.py
    ```

## How It Works

The agent uses the **Constant-Alpha Monte Carlo Update Rule** to estimate the Action-Value function $Q(S, A)$. 
This approach allows the agent to slowly "forget" old, suboptimal experiences and focus on recent, improved strategies.

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [G_t - Q(S_t, A_t)]$$

Where:
* $\alpha$ is the learning rate (e.g., 0.01).
* $G_t$ is the total return of the episode.
* The term $[G_t - Q(S_t, A_t)]$ represents the prediction error.
## New Features

### 3:2 Payout
The game now rewards **"Natural Blackjacks"** (getting 21 with the first two cards) with a **3:2 payout** (1.5x return).

### Gamble Night Simulation
The project now includes a **"Gamble Night"** mode which runs after training.
* The agent starts with a bankroll (e.g., **500 Euro**).
* It places fixed bets (e.g., **10 Euro**) on every hand.
* The simulation tracks the bankroll over time, simulating a real night at the casino.

## TODO

* Heatmap
* Double Down
* Real Deck with 4 Colors
