import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def calculate_metrics(results):
    """Calculate performance metrics"""
    # Annual return
    total_days = len(results)
    total_return = (results['portfolio_value'].iloc[-1] / results['portfolio_value'].iloc[0]) - 1
    annual_return = ((1 + total_return) ** (252/total_days) - 1) * 100

    # Daily returns volatility
    daily_vol = results['strategy_returns'].std() * np.sqrt(252)

    # Sharpe ratio
    risk_free_rate = 0.02  # Assuming 2% risk-free rate
    excess_returns = results['strategy_returns'] - risk_free_rate/252
    sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()

    # Maximum drawdown
    rolling_max = results['portfolio_value'].cummax()
    drawdowns = (results['portfolio_value'] - rolling_max) / rolling_max
    max_drawdown = drawdowns.min() * 100

    # Win rate
    trades = results[results['trade'].notna() & (results['trade'] != 0)]
    winning_trades = trades[trades['strategy_returns'] > 0]
    win_rate = (len(winning_trades) / len(trades)) * 100 if len(trades) > 0 else 0

    return {
        'annual_return': annual_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'volatility': daily_vol * 100
    }

def plot_drawdown(results):
    """Plot drawdown over time"""
    rolling_max = results['portfolio_value'].cummax()
    drawdowns = (results['portfolio_value'] - rolling_max) / rolling_max * 100

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=results.index,
        y=drawdowns,
        fill='tozeroy',
        name='Drawdown',
        line=dict(color='red')
    ))

    fig.update_layout(
        title='Portfolio Drawdown',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        template='plotly_white',
        showlegend=True,
        yaxis=dict(tickformat='.1f')
    )

    return fig

def plot_equity_curve(results):
    """Plot equity curve"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=results.index,
        y=results['portfolio_value'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#17a2b8')
    ))

    # Add buy and hold comparison
    initial_price = results['Close'].iloc[0]
    initial_shares = results['portfolio_value'].iloc[0] / initial_price
    buy_hold = initial_shares * results['Close']

    fig.add_trace(go.Scatter(
        x=results.index,
        y=buy_hold,
        mode='lines',
        name='Buy & Hold',
        line=dict(color='#666666', dash='dash')
    ))

    fig.update_layout(
        title='Portfolio Value Over Time vs Buy & Hold',
        xaxis_title='Date',
        yaxis_title='Portfolio Value ($)',
        template='plotly_white'
    )

    return fig

def plot_trades(data, results):
    """Plot trading signals"""
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Plot price
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#666666')
        ),
        secondary_y=False
    )

    # Plot moving averages
    fig.add_trace(
        go.Scatter(
            x=results.index,
            y=results['SMA_short'],
            mode='lines',
            name=f'{results.SMA_short.name} MA',
            line=dict(color='#17a2b8')
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=results.index,
            y=results['SMA_long'],
            mode='lines',
            name=f'{results.SMA_long.name} MA',
            line=dict(color='#28a745')
        ),
        secondary_y=False
    )

    # Plot buy signals
    buy_signals = results[results['trade'] > 0]
    fig.add_trace(
        go.Scatter(
            x=buy_signals.index,
            y=buy_signals['Close'],
            mode='markers',
            name='Buy',
            marker=dict(
                color='green',
                size=10,
                symbol='triangle-up'
            )
        ),
        secondary_y=False
    )

    # Plot sell signals
    sell_signals = results[results['trade'] < 0]
    fig.add_trace(
        go.Scatter(
            x=sell_signals.index,
            y=sell_signals['Close'],
            mode='markers',
            name='Sell',
            marker=dict(
                color='red',
                size=10,
                symbol='triangle-down'
            )
        ),
        secondary_y=False
    )

    # Add MA difference percentage on secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=results.index,
            y=results['ma_diff_pct'],
            mode='lines',
            name='MA Difference %',
            line=dict(color='#ffc107', dash='dot'),
            opacity=0.5
        ),
        secondary_y=True
    )

    fig.update_layout(
        title='Trading Signals',
        xaxis_title='Date',
        template='plotly_white',
        height=600
    )

    # Update y-axes labels
    fig.update_yaxes(title_text="Price ($)", secondary_y=False)
    fig.update_yaxes(title_text="MA Difference (%)", secondary_y=True)

    return fig