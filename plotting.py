import plotly.express as px


def plot_tsne(projections, batches, name):
    """
    Scatter-plot 2D projections colored by batch/label and save as an HTML file.

    Args:
        projections (array-like): 2D coordinates, shape (n_samples, 2).
        batches (array-like): Per-sample color values (e.g. batch or confidence labels).
        name (str): Output HTML file path.
    """
    fig = px.scatter(
        projections, x=0, y=1,
        color=batches, labels={'color': 'batches'}
    )
    fig.write_html(name)
    return
