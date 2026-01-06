import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# --- 1. The Environment ---

class BlackjackEnvironment:
    def __init__(self):
        # Card values: 2-10, J/Q/K=10, A=1 or 11
        # We model the deck as infinite (drawing with replacement)
        self.deck = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]

    def draw_card(self):
        return random.choice(self.deck)

    def draw_hand(self):
        return [self.draw_card(), self.draw_card()]

    def usable_ace(self, hand):
        """Returns True if the hand has a usable Ace (Ace counted as 11)."""
        return 11 in hand and sum(hand) <= 21

    def sum_hand(self, hand):
        """Calculates the sum of the hand. Adjusts Aces if the sum > 21."""
        if self.usable_ace(hand):
            return sum(hand)
        total = sum(hand)
        while total > 21 and 11 in hand:
            hand[hand.index(11)] = 1
            total = sum(hand)
        return total

    def get_obs(self):
        """Returns the current state: (Player Sum, Dealer Visible Card, Usable Ace, Can Double)"""
        return (self.sum_hand(self.player_hand), self.dealer_hand[0], self.usable_ace(self.player_hand), len(self.player_hand) == 2)

    def reset(self):
        self.player_hand = self.draw_hand()
        self.dealer_hand = self.draw_hand()
        return self.get_obs()

    def step(self, action):
        """
        Action 0: Stick
        Action 1: Hit
        Action 2: Double Down
        Returns: (next_state, reward, done)
        """
        # --- Player's Turn ---
        if action == 1:  # Hit
            self.player_hand.append(self.draw_card())
            if self.sum_hand(self.player_hand) > 21:
                return self.get_obs(), -1, True # Bust
            else:
                return self.get_obs(), 0, False # Continue game
        
        elif action == 2: # Double Down
            if len(self.player_hand) != 2:
                # Invalid move (trying to double after hitting) - Heavy Penalty and End
                return self.get_obs(), -10, True 
            
            # Double Down: Draw one card only, then forced Stick
            self.player_hand.append(self.draw_card())
            
            if self.sum_hand(self.player_hand) > 21:
                return self.get_obs(), -2, True # Bust (Double Loss)
            
            # Forced Stick check against dealer
            # (Fall through to dealer logic below, but we need to set reward scale)
            reward_mult = 2
            
            # --- Dealer's Turn (Code shared/duplicated for clarity) ---
            decision_made = False
            while not decision_made:
                dealer_sum = self.sum_hand(self.dealer_hand)
                if dealer_sum < 17:
                    self.dealer_hand.append(self.draw_card())
                else:
                    decision_made = True
            
            player_sum = self.sum_hand(self.player_hand)
            dealer_sum = self.sum_hand(self.dealer_hand)
            
            if dealer_sum > 21:
                return self.get_obs(), 1 * reward_mult, True # Win Double
            
            if player_sum > dealer_sum:
                return self.get_obs(), 1 * reward_mult, True # Win Double
            elif player_sum < dealer_sum:
                return self.get_obs(), -1 * reward_mult, True # Lose Double
            else:
                return self.get_obs(), 0, True # Draw

        else:  # Stick (Action 0)
            # --- Dealer's Turn ---
            decision_made = False
            while not decision_made:
                dealer_sum = self.sum_hand(self.dealer_hand)
                if dealer_sum < 17:
                    self.dealer_hand.append(self.draw_card())
                else:
                    decision_made = True
            
            player_sum = self.sum_hand(self.player_hand)
            dealer_sum = self.sum_hand(self.dealer_hand)
            
            # Compare hands
            if dealer_sum > 21:
                # Dealer busts, Player wins
                reward = 1
                if player_sum == 21 and len(self.player_hand) == 2:
                    reward = 1.5
                elif len(self.player_hand) > 2:
                    reward = 2.5 # Bonus for hitting and winning (Aggressive) - Increased from 1.5
                return self.get_obs(), reward, True
            
            if player_sum > dealer_sum:
                # Win
                reward = 1
                if player_sum == 21 and len(self.player_hand) == 2:
                    reward = 1.5
                elif len(self.player_hand) > 2:
                    reward = 2.5 # Bonus for hitting and winning (Aggressive) - Increased from 1.5
                return self.get_obs(), reward, True
            elif player_sum < dealer_sum:
                return self.get_obs(), -1, True # Lose
            else:
                return self.get_obs(), 0, True # Draw

# --- 2. The Agent ---

class MonteCarloAgent:
    def __init__(self, action_space=[0, 1, 2], alpha=0.01, gamma=1.0, epsilon=1.0):
        self.Q = {} # Dictionary mapping (state, action) -> value
        self.action_space = action_space
        self.alpha = alpha # Learning rate
        self.gamma = gamma # Discount factor 1 = no discount
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99995
        
    def get_q(self, state, action):
        return self.Q.get((state, action), 0.0)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.action_space)
        else:
            # Greedy choice
            q_values = [self.get_q(state, a) for a in self.action_space]
            # Break ties randomly
            max_q = max(q_values)
            best_actions = [a for a, q in zip(self.action_space, q_values) if q == max_q]
            return random.choice(best_actions)

    def update(self, episode):
        # Episode is a list of (state, action, reward)
        # G = sum(gamma^k * R)
        
        G = 0
        # Traverse backward
        for state, action, reward in reversed(episode):
            G = self.gamma * G + reward
            
            # Update Q-value
            old_q = self.get_q(state, action)
            # Q[S,A] = Q[S,A] + alpha * (G - Q[S,A])
            self.Q[(state, action)] = old_q + self.alpha * (G - old_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_optimal_policy(self):
        """
        Returns a dictionary mapping state -> best action (0 or 1).
        """
        policy = {}
        # Identify all unique states visited
        states = set(state for (state, action) in self.Q.keys())
        
        for state in states:
            q_values = [self.get_q(state, a) for a in self.action_space]
            # Choose the action with the highest Q-value
            best_action = self.action_space[np.argmax(q_values)]
            policy[state] = best_action
            
        return policy

def gamble_night(agent, env, bankroll=500, bet=10, max_hands=100):
    print(f"\n--- Gamble Night Simulation ---")
    print(f"Starting Bankroll: {bankroll} Euro")
    print(f"Bet per hand: {bet} Euro")
    
    agent.epsilon = 0 # Play optimally
    
    current_bankroll = bankroll
    history = []
    
    for i in range(max_hands):
        if current_bankroll < bet:
            print("Broke! Game Over.")
            break
            
        state = env.reset()
        done = False
        actions_taken = []
        
        # print(f"Hand {i+1}: {state}")
        
        while not done:
            action = agent.choose_action(state)
            action_name = 'Stick'
            if action == 1: action_name = 'Hit'
            if action == 2: action_name = 'Double'
            actions_taken.append(action_name)
            
            # print(f"Action: {action_name}")
            state, reward, done = env.step(action)
        
        # Reward is 1, 1.5, 0, or -1.
        winnings = reward * bet
        current_bankroll += winnings
        history.append(current_bankroll)
        
        outcome = "Draw"
        if reward > 0: outcome = "Win"
        elif reward < 0: outcome = "Loss"
        
        # Calculate effective bet for display (Double Down = 2x)
        effective_bet = bet * 2 if (action == 2) else bet
        
        action_str = ", ".join(actions_taken)
        print(f"Hand {i+1}: {outcome} ({action_str}). Player: {env.player_hand} Dealer: {env.dealer_hand}. Bankroll: {current_bankroll:.1f} Euro")

    print(f"\n--- Night Over ---")
    print(f"Final Bankroll: {current_bankroll:.1f} Euro")
    print(f"Profit/Loss: {current_bankroll - bankroll:.1f} Euro")

# --- 3. Visualization ---

def plot_policy_and_value(agent):
    # Dimensions
    # Usable Ace: Player sum 12-21 (10 rows)
    player_sums_usable = range(21, 11, -1) 
    # No Usable Ace: Player sum 4-21 (18 rows)
    player_sums_no_usable = range(21, 3, -1)
    
    dealer_cards = range(2, 12) # 2-10, Ace(11)
    dealer_labels = [2,3,4,5,6,7,8,9,10,'A']
    
    # Init matrices
    val_use = np.zeros((len(player_sums_usable), len(dealer_cards)))
    pol_use = np.zeros((len(player_sums_usable), len(dealer_cards)))
    txt_use = np.empty((len(player_sums_usable), len(dealer_cards)), dtype=object)
    
    val_no = np.zeros((len(player_sums_no_usable), len(dealer_cards)))
    pol_no = np.zeros((len(player_sums_no_usable), len(dealer_cards)))
    txt_no = np.empty((len(player_sums_no_usable), len(dealer_cards)), dtype=object)

    action_labels = {0: 'S', 1: 'H', 2: 'DD'}

    # Fill Usable Ace
    for i, p_sum in enumerate(player_sums_usable):
        for j, d_card in enumerate(dealer_cards):
            state = (p_sum, d_card, True, True)
            q_values = [agent.get_q(state, a) for a in agent.action_space]
            val_use[i, j] = max(q_values)
            best_a = np.argmax(q_values)
            pol_use[i, j] = best_a
            txt_use[i, j] = action_labels[best_a]

    # Fill No Usable Ace
    for i, p_sum in enumerate(player_sums_no_usable):
        for j, d_card in enumerate(dealer_cards):
            state = (p_sum, d_card, False, True)
            q_values = [agent.get_q(state, a) for a in agent.action_space]
            val_no[i, j] = max(q_values)
            best_a = np.argmax(q_values)
            pol_no[i, j] = best_a
            txt_no[i, j] = action_labels[best_a]

    # Custom Color Map for Policy: Stick(0)=Green, Hit(1)=Yellow, Double(2)=Red
    cmap_policy = ListedColormap(['green', 'yellow', 'red'])

    # Labels for Usable Ace (e.g., 21 -> "A, 10")
    usable_ace_labels = [f"A, {s-11}" if s != 12 else "A, A" for s in player_sums_usable]

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # helper
    def plot_heatmap(ax, data, title, y_labels, annot=True, cmap="viridis", vmin=None, vmax=None, cbar=True, fmt=""):
        sns.heatmap(data, ax=ax, cmap=cmap, annot=annot, fmt=fmt, 
                    xticklabels=dealer_labels, yticklabels=y_labels,
                    vmin=vmin, vmax=vmax, cbar=cbar)
        ax.set_title(title)
        ax.set_ylabel("Player Hand")
        ax.set_xlabel("Dealer Showing")
        ax.set_yticklabels(y_labels, rotation=0)

    # 1. Value (Usable Ace)
    plot_heatmap(axes[0, 0], val_use, "Value - Usable Ace", usable_ace_labels, annot=True, fmt=".3f")
    
    # 2. Policy (Usable Ace)
    plot_heatmap(axes[0, 1], pol_use, "Policy - Usable Ace (S/H/DD)", usable_ace_labels, 
                 annot=txt_use, cmap=cmap_policy, cbar=False, vmin=0, vmax=2)
    
    # 3. Value (No Usable Ace)
    plot_heatmap(axes[1, 0], val_no, "Value - No Usable Ace", player_sums_no_usable, annot=True, fmt=".3f")

    # 4. Policy (No Usable Ace)
    plot_heatmap(axes[1, 1], pol_no, "Policy - No Usable Ace (S/H/DD)", player_sums_no_usable, 
                 annot=txt_no, cmap=cmap_policy, cbar=False, vmin=0, vmax=2)

    plt.tight_layout()
    plt.show()

# --- 4. Training Loop ---

if __name__ == "__main__":
    env = BlackjackEnvironment()
    agent = MonteCarloAgent()

    num_episodes = 500000
    
    # Tracking for visualization
    win_count = 0
    draw_count = 0
    loss_count = 0
    
    print(f"Starting training for {num_episodes} episodes...")

    for i in range(num_episodes):
        state = env.reset()
        done = False
        episode = []
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            
        agent.update(episode)
        agent.decay_epsilon()
        
        # Update counters based on *Last Reward*
        final_reward = episode[-1][2]
        if final_reward >= 1: # Win
            win_count += 1
        elif final_reward <= -1: # Loss
            loss_count += 1
        else: # Draw (0)
            draw_count += 1
            
        if (i+1) % 50000 == 0:
            total_in_batch = 50000
            print(f"Episode {i+1}/{num_episodes} - "
                  f"Win: {win_count/total_in_batch:.2f} | "
                  f"Draw: {draw_count/total_in_batch:.2f} | "
                  f"Loss: {loss_count/total_in_batch:.2f} | "
                  f"Epsilon: {agent.epsilon:.4f}")
            win_count = 0
            draw_count = 0
            loss_count = 0

    print("Training finished.")

    # --- Gamble Night ---
    gamble_night(agent, env)
    
    # --- Visualize ---
    plot_policy_and_value(agent)
