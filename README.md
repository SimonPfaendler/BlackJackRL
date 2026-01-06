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
| **Actions** | `0` (Stick), `1` (Hit) `2` (Double Down) |
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
    pip install numpy matplotlib seaborn
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

## Simulation Results

### Policy & Value Visualization Heatmap
<img width="1920" height="967" alt="Figure_3" src="https://github.com/user-attachments/assets/8809e600-916d-42da-a213-537e0f442f0f" />


The visualization above shows the **Value Function** (expected return) and **Optimal Policy** learned by the agent.
- **Key Learnings:** The agent has correctly learned to **Double Down (DD)** on 11,10,9 maximizing potential payout.
- **Current Behavior:** The agent is currently quite conservative and **does not Hit (H) very often**, preferring to Stick (S) on most hands.

### Training Win Rates (500,000 Episodes)
```text
Episode 50000/500000 - Win: 0.37 | Draw: 0.06 | Loss: 0.57 | Epsilon: 0.0821
...
Episode 500000/500000 - Win: 0.42 | Draw: 0.09 | Loss: 0.50 | Epsilon: 0.0100
Training finished.
```

### Gamble Night Example
```text
Starting Bankroll: 500 Euro
Bet per hand: 10 Euro
Hand 1: Win (Hit, Hit, Hit, Hit, Hit, Stick). Player: [2, 1, 3, 1, 6, 2, 3] Dealer: [7, 9, 6]. Bankroll: 525.0 Euro
Hand 2: Loss (Hit). Player: [2, 10, 10] Dealer: [9, 10]. Bankroll: 515.0 Euro
...
Hand 20: Loss (Hit, Hit, Stick). Player: [4, 2, 5, 4] Dealer: [10, 4, 6]. Bankroll: 560.0 Euro
...
Hand 27: Win (Double). Player: [3, 8, 10] Dealer: [4, 7, 6]. Bankroll: 600.0 Euro
...
Hand 100: Loss (Stick). Player: [10, 4] Dealer: [10, 10]. Bankroll: 1020.0 Euro

--- Night Over ---
Final Bankroll: 1020.0 Euro
Profit/Loss: 520.0 Euro
```

## TODO

* Real Deck with 4 Colors
* Improve Hit Behavior of Agent
* Gamble Night Vizualization Episodes
* Add Split maybe?
