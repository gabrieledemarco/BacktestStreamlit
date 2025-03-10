import numpy as np
import pandas as pd

def monte_carlo_simulation(returns, leverage_values, position_sizes, simulations=1000):
    """
    Testa diverse combinazioni di leva e position sizing con simulazioni Monte Carlo.

    Args:
        returns (pd.Series): Serie storica dei rendimenti.
        leverage_values (list): Lista di leverage da testare.
        position_sizes (list): Lista di frazioni di capitale da investire.
        simulations (int): Numero di simulazioni.

    Returns:
        dict: Risultati con capitali finali medi per ogni combinazione.
    """
    results = {}

    for lev in leverage_values:
        for pos_size in position_sizes:
            final_capitals = []
            for _ in range(simulations):
                capital = 10000  # Capitale iniziale
                for r in np.random.choice(returns, size=len(returns), replace=True):
                    capital += capital * lev * pos_size * r  # Simula trade con leverage e frazione capitale
                    capital = max(capital, 0)  # Evita valori negativi (margin call)
                final_capitals.append(capital)

            results[(lev, pos_size)] = np.mean(final_capitals)  # Capitale medio finale

    return results


# Testiamo combinazioni di leva e frazioni di capitale
leverage_values = [1, 2, 3, 4, 5]  # Testiamo leverage tra 1x e 5x
position_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]  # Testiamo frazioni tra 10% e 50%
returns = pd.Series(np.random.normal(0.001, 0.02, 252))

# Eseguiamo la simulazione
results = monte_carlo_simulation(returns, leverage_values, position_sizes)

# Ordiniamo i risultati e mostriamo i migliori
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
print("Top 5 combinazioni ottimali:")
for (lev, pos_size), final_capital in sorted_results[:5]:
    print(f"Leverage: {lev}x, Position Size: {pos_size:.1%}, Capitale Finale Medio: {final_capital:.2f}")
