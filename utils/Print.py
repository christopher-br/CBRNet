# Import necessary modules
import numpy as np

import plotly.graph_objects as go

def summary_plot(true_dr, 
                 mlp_dr, drnet_dr, cbrnet_dr, linreg_dr, gps_dr, vcnet_dr, 
                 mise_mlp, mise_drnet, mise_cbrnet, mise_linreg, mise_gps, mise_vcnet, 
                 dosages, clusters, s_bias, d_bias):
    """
    Function to generate summary plot for logging in W&B
    
    Parameters:
        true_dr (np array): An np array of the true dose response
        ...
        mlp_mise (float): The MISE of the MLP predictions
        ...
        dosages (np array): All dosages of the tranining observations
        clusters (np array): All clusters of the training observations
        bias (float): The bias strength in the training data
    """
    # Get bins and treatment strengths
    samples_power_of_two = 6
    num_integration_samples = 2 ** samples_power_of_two + 1
    bins = np.linspace(0, 1, num_integration_samples + 1)
    yaxis = np.linspace(0, 1, num_integration_samples)
    
    # Get histograms per cluster
    bars_c0 = np.histogram(dosages[clusters==0], bins)[0]
    bars_c1 = np.histogram(dosages[clusters==1], bins)[0]
    bars_c2 = np.histogram(dosages[clusters==2], bins)[0]
    
    # Standardize histograms
    max_bar = np.max((bars_c0, bars_c1, bars_c2))
    max_dr = np.max((true_dr, mlp_dr, drnet_dr, cbrnet_dr, linreg_dr, gps_dr, vcnet_dr))
    bars_c0 = (bars_c0 / max_bar) * max_dr
    bars_c1 = (bars_c1 / max_bar) * max_dr
    bars_c2 = (bars_c2 / max_bar) * max_dr
    
    # Build figure
    figure = go.Figure(layout=go.Layout(
        annotations=[
            # Add textbox with MISEs
            go.layout.Annotation(
                text='MISE:<br>MLP: <br>DRNet: <br>LinReg: <br>GPS: <br>VCNet: <br>CBRNet: ',
                align='left',
                showarrow=False,
                xref='paper',
                yref='paper',
                x=1.05,
                y=0.0,
                borderwidth=1),
            go.layout.Annotation(
                text=' <br>'+ '{:.4}'.format(mise_mlp) +'<br>'+ '{:.4}'.format(mise_drnet) +'<br>'+ '{:.4}'.format(mise_linreg) +'<br>'+'{:.4}'.format(mise_gps) +'<br>'+'{:.4}'.format(mise_vcnet) +'<br>'+ '{:.4}'.format(mise_cbrnet),
                align='left',
                showarrow=False,
                xref='paper',
                yref='paper',
                x=1.08,
                y=0.0,
                borderwidth=1)
            ]
        )
                       )
    figure.add_trace(go.Scatter(x=yaxis, y=true_dr, line_shape='linear', name='Truth'))
    figure.add_trace(go.Scatter(x=yaxis, y=mlp_dr, line_shape='linear', name='MLP'))
    figure.add_trace(go.Scatter(x=yaxis, y=drnet_dr, line_shape='linear', name='DRNets'))
    figure.add_trace(go.Scatter(x=yaxis, y=linreg_dr, line_shape='linear', name='LinReg'))
    figure.add_trace(go.Scatter(x=yaxis, y=gps_dr, line_shape='linear', name='GPS'))
    figure.add_trace(go.Scatter(x=yaxis, y=vcnet_dr, line_shape='linear', name='VCNet'))
    figure.add_trace(go.Scatter(x=yaxis, y=cbrnet_dr, line_shape='linear', name=' CBRNets'))
    figure.add_trace(go.Bar(x=yaxis, y=bars_c0, name="Cluster 0", opacity=0.5, marker_color='lightblue'))
    figure.add_trace(go.Bar(x=yaxis, y=bars_c1, name="Cluster 1", opacity=0.5, marker_color='lightgreen'))
    figure.add_trace(go.Bar(x=yaxis, y=bars_c2, name="Cluster 2", opacity=0.5, marker_color='lightpink'))
    
    # Update layout
    figure.update_layout(
        barmode='group', 
        bargap=0.0, 
        bargroupgap=0.0, 
        paper_bgcolor='white', 
        plot_bgcolor='white',
        shapes=[go.layout.Shape(type='rect', 
                                xref='paper', yref='paper', 
                                x0=0.0, y0=0.0, 
                                x1=1.0, y1=1.0,
                                line={'width': 1, 'color': 'black'})],
        title=('Summary | Selec. bias: ' + str(s_bias) + ' Distr. bias: ' + str(d_bias)),
        xaxis_title='Dosage',
        yaxis_title='Response / Relative Frequency',
        legend_title='Legend')
    
    return figure