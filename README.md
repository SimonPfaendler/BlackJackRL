# ♠️ Blackjack Reinforcement Learning Agent

A Python implementation of a Reinforcement Learning agent that learns to play a simplified version of Blackjack using **Monte Carlo First-Visit methods**.

## Overview

This project builds a custom Blackjack environment and an RL agent from scratch. The agent treats the game as a **Markov Decision Process (MDP)** and learns the optimal strategy (Policy) through trial and error over 5,000,000 episodes.

For simplicity and faster convergence, this model uses a **"One Color"** deck (ignoring suits), focusing purely on card values and probability.

### Key Concepts
* **Monte Carlo Method:** The agent learns from complete episodes (games played to the end) rather than bootstrapping.
* **First-Visit MC:** The Q-value is updated using the return from the *first* time a state was visited in an episode.
* **Epsilon-Greedy Policy:** The agent explores random moves 10% of the time to ensure it discovers new strategies, while exploiting its best knowledge 90% of the time.

## ⚙️ The Environment

The environment is a simplified simulation of Blackjack:

| Component | Description |
| :--- | :--- |
| **State Space** | `(Player Sum, Dealer Showing Card, Usable Ace)` |
| **Actions** | `0` (Stick), `1` (Hit) |
| **Rewards** | `+1` (Win), `-1` (Loss/Bust), `0` (Draw) |
| **Simplification** | Infinite deck, single suit (uniform probability for 2-9, 4x for 10s). |

## 🛠️ Installation & Usage

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

## 🧠 How It Works

The agent uses the **Monte Carlo Update Rule** to estimate the Action-Value function $Q(S, A)$.

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \frac{1}{N(S_t, A_t)} [G_t - Q(S_t, A_t)]$$
