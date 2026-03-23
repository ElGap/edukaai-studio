"""
Visualizations Module

Creates charts and graphs for training metrics.
"""

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not available. Charts will be disabled.")


def create_loss_chart(train_losses: dict, val_losses: dict, best_iter: int = None):
    """
    Create an interactive loss chart.
    
    Args:
        train_losses: Dictionary of {iteration: loss}
        val_losses: Dictionary of {iteration: val_loss}
        best_iter: Best iteration (marked with star)
        
    Returns:
        plotly Figure or None if plotly not available
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Training & Validation Loss', 'Generalization Gap'),
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4]
    )
    
    # Sort iterations
    train_iters = sorted(train_losses.keys())
    val_iters = sorted(val_losses.keys())
    
    # Training loss line
    if train_iters:
        fig.add_trace(
            go.Scatter(
                x=train_iters,
                y=[train_losses[i] for i in train_iters],
                mode='lines',
                name='Train Loss',
                line=dict(color='blue', width=2),
                opacity=0.6
            ),
            row=1, col=1
        )
    
    # Validation loss line
    if val_iters:
        val_x = val_iters
        val_y = [val_losses[i] for i in val_iters]
        
        # Mark best iteration
        marker_colors = ['red' if i == best_iter else 'green' for i in val_x]
        marker_sizes = [15 if i == best_iter else 8 for i in val_x]
        
        fig.add_trace(
            go.Scatter(
                x=val_x,
                y=val_y,
                mode='lines+markers',
                name='Val Loss',
                line=dict(color='green', width=2),
                marker=dict(
                    color=marker_colors,
                    size=marker_sizes,
                    symbol='star' if best_iter else 'circle'
                )
            ),
            row=1, col=1
        )
    
    # Generalization gap
    if train_iters and val_iters:
        gap_iters = [i for i in val_iters if i in train_losses]
        gaps = [val_losses[i] - train_losses[i] for i in gap_iters]
        
        if gaps:
            fig.add_trace(
                go.Scatter(
                    x=gap_iters,
                    y=gaps,
                    mode='lines',
                    name='Generalization Gap',
                    line=dict(color='orange', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(255,165,0,0.2)'
                ),
                row=2, col=1
            )
    
    # Update layout
    fig.update_layout(
        height=600,
        showlegend=True,
        template='plotly_white',
        title_text="Training Metrics",
        title_x=0.5
    )
    
    # Update axes
    fig.update_xaxes(title_text="Iteration", row=1, col=1)
    fig.update_xaxes(title_text="Iteration", row=2, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Gap (Val - Train)", row=2, col=1)
    
    # Add annotation for best iteration
    if best_iter and best_iter in val_losses:
        fig.add_annotation(
            x=best_iter,
            y=val_losses[best_iter],
            text="⭐ BEST",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-40,
            row=1, col=1
        )
    
    return fig


def create_simple_loss_plot(val_losses: dict):
    """
    Create a simple matplotlib-like text chart (fallback when plotly not available).
    
    Args:
        val_losses: Dictionary of {iteration: val_loss}
        
    Returns:
        str: ASCII art chart
    """
    if not val_losses:
        return "No data available"
    
    iters = sorted(val_losses.keys())
    losses = [val_losses[i] for i in iters]
    
    # Simple ASCII chart
    lines = ["Validation Loss Trend:", "-" * 40]
    
    min_loss = min(losses)
    max_loss = max(losses)
    range_loss = max_loss - min_loss if max_loss != min_loss else 1
    
    for i, loss in zip(iters, losses):
        # Create bar
        bar_len = int(20 * (max_loss - loss) / range_loss)
        bar = "█" * bar_len
        lines.append(f"{i:4d} │{bar:20s}│ {loss:.4f}")
    
    lines.append("-" * 40)
    return "\n".join(lines)


def format_metrics_summary(train_losses: dict, val_losses: dict) -> str:
    """
    Format metrics as a text summary.
    
    Args:
        train_losses: Training losses
        val_losses: Validation losses
        
    Returns:
        str: Formatted summary
    """
    summary = []
    
    if train_losses:
        train_vals = list(train_losses.values())
        summary.append(f"📈 Training Loss:")
        summary.append(f"   Initial: {train_vals[0]:.4f}")
        summary.append(f"   Final: {train_vals[-1]:.4f}")
        summary.append(f"   Min: {min(train_vals):.4f}")
    
    if val_losses:
        val_vals = list(val_losses.values())
        summary.append(f"📊 Validation Loss:")
        summary.append(f"   Best: {min(val_vals):.4f} at iter {min(val_losses.keys(), key=lambda k: val_losses[k])}")
        summary.append(f"   Final: {val_vals[-1]:.4f}")
    
    if train_losses and val_losses:
        # Find common iterations
        common = set(train_losses.keys()) & set(val_losses.keys())
        if common:
            gaps = [val_losses[i] - train_losses[i] for i in common]
            avg_gap = sum(gaps) / len(gaps)
            summary.append(f"🔄 Generalization Gap: {avg_gap:.4f}")
    
    return "\n".join(summary)
