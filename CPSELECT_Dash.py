# inspired by https://dash.plotly.com/interactive-graphing

from dash import Dash, dcc, html, Input, Output, callback, State, no_update
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import h5py
import numpy as np
import functools
import time
import datetime
import math


from changepoynt.algorithms.sst import SST
from changepoynt.algorithms.esst import ESST

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll',
        'color': 'black'
    },
    'div': {
        'padding': '.3rem',
        'width': '9'
                 '0%',
        'margin': 'auto',
        'boxShadow': 'dimgrey 4px 4px 2px',
        'border-radius': '10px',
        'backgroundColor': 'white',
        'marginTop': '1rem',
    },
    'dropdown': {
        'margin': 'auto',
        'width': '50%',
        'border-radius': '10px',
        'color': 'black'
    }
}


@functools.cache
def read_data():
    # time the read (and use a print to make sure the cache works)
    start = time.perf_counter()

    # read the pressure data -------------------------------------------------------------------------------------------

    # get the data into memory as a dict of numpy arrays
    file_path = "./HS2/Pressure_monitoring/HS2_Pressure_Monitoring.mat"
    with h5py.File(file_path) as f:
        dat = {k: np.array(v) for k, v in f.items()}

    # time array for inj dat
    Nt = len(dat['time_inj'][0])
    t_inj_dat = np.zeros(Nt)
    for i in range(Nt):
        t_inj_dat[i] = datetime.timedelta(dat['time_inj'][0, i] - dat['time_inj'][0, 0]).seconds
    t_inj_dat /= 3600

    # get the timings from the recordings
    t0 = dat['time_inj'][0, 0]
    tend = dat['time_inj'][0, -1]

    # read the flow data -----------------------------------------------------------------------------------------------
    file_path = "./HS2/Injection_protocol/20170208_HS2inj.dat"

    vol_dat = pd.read_csv(file_path)
    id0_vol = np.argmin(np.abs(vol_dat['time_datenum'] - t0))
    idend_vol = np.argmin(np.abs(vol_dat['time_datenum'] - tend))
    Nt = len(vol_dat['time_datenum'][id0_vol:idend_vol])
    t_vol = np.zeros(Nt)
    for i in range(id0_vol, idend_vol):
        t_vol[i - id0_vol] = datetime.timedelta(vol_dat['time_datenum'][i] - vol_dat['time_datenum'][id0_vol]).seconds

    # get the flow and pressure signal of interest
    flow_time = t_vol / 3600
    flow_signal = vol_dat['cleaned_flow_lpm'][id0_vol:idend_vol].to_numpy()
    pressure_time = t_inj_dat[:-1]
    pressure_signal = dat['pressure'][0][:-1]

    # end the time
    print(f"Data is in memory (and cached) and loading took: {time.perf_counter() - start} s.")
    return flow_time, flow_signal, pressure_time, pressure_signal


def make_signals():

    # get the raw data
    flow_time, flow_signal, pressure_time, pressure_signal = read_data()

    # Create a figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # make the plot
    fig.add_trace(
        go.Scatter(x=flow_time, y=flow_signal, name="Flow Signal", line=dict(color="orange")),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=pressure_time, y=pressure_signal, name="Pressure Signal", line=dict(color="green")),
        secondary_y=True,
    )
    return fig


def update_marks(figure, v_line: float = None, selected_points: list[dict] = None, first_name: str = "Flow Signal", second_name: str = "Pressure Signal"):

    # get the traces we want to change to load the data from
    traces_data = [trace for trace in figure['data'] if 'name' in trace and trace['name'] in [first_name, second_name]]

    # delete traces that are unwished for
    figure['data'] = [trace for trace in figure['data'] if 'name' not in trace or not trace['name'].startswith('Sel.')]

    # check whether we have shapes (such as the line) and delete those as well
    if 'shapes' in figure['layout']:
        figure['layout']['shapes'] = [shape for shape in figure['layout']['shapes'] if 'name' not in shape or shape['name'] != 'Sel. Time']

    # create the figure object from the serialized json dict
    figure = go.Figure(figure)

    # get the raw data from the figure
    flow_time = np.array(traces_data[0]['x'])
    flow_signal = np.array(traces_data[0]['y'])
    flow_name = traces_data[0]['name'].split(' ')[0]
    if len(traces_data) > 1:
        pressure_time = np.array(traces_data[1]['x'])
        pressure_signal = np.array(traces_data[1]['y'])
        pressure_name = traces_data[1]['name'].split(' ')[0]

    # draw the new lines
    if v_line is not None and np.min(flow_time) <= v_line <= np.max(flow_time):
        figure.add_vline(x=v_line, name="Sel. Time")
    if selected_points:
        # get the coordinates
        maxt = np.max(flow_time)
        mint = np.min(flow_time)
        coords = [np.where(flow_time == point)[0][0] for point in selected_points if mint <= point <= maxt]

        # mark the points on both signals
        figure.add_trace(
            go.Scatter(x=flow_time[coords], y=flow_signal[coords], name=f"Sel. {flow_name}", mode='markers', yaxis='y1'),
        )
        if len(traces_data) > 1:
            figure.add_trace(
                go.Scatter(x=pressure_time[coords], y=pressure_signal[coords], name=f"Sel. {pressure_name}", mode='markers', yaxis='y2'),
            )
    return figure


app.layout = html.Div([
    html.Div(children=[
        html.H1(f'Change Point Scatter Plot',
                style={'fontSize': 40},
                id='header'),
    ],
        style=styles['div']
    ),
    html.Div(children=[
        dcc.Graph(
            id='signal-graph',
            figure=make_signals()
        ),
    ],
        style=styles['div']
    ),
    html.Div(children=[
        html.Div(children=[
            html.Div(children=[
                dcc.Loading(id="loading-1",
                            type="default",
                            children=[
                                dcc.Graph(
                                    id='change-scatter',
                                    figure={},
                                    style={'width': '100%', 'aspect-ratio': "1/1", "transform": "translateY(25%)"}
                                )
                            ],
                            ),
            ],
                className='four columns'
            ),
            html.Div(children=[
                html.Div(children=[
                    html.Div(children=[
                        dcc.Loading(id="loading-flow-change",
                                    type="default",
                                    children=[
                                        dcc.Graph(
                                            id='flow-change-plot',
                                            figure={},
                                        )],
                                    ),
                    ],
                        className="twelve columns"
                    ),
                ],
                    className="row",
                ),
                html.Div(children=[
                    html.Div(children=[
                        dcc.Loading(id="loading-pressure-change",
                                    type="default",
                                    children=[
                                        dcc.Graph(
                                            id='pressure-change-plot',
                                            figure={},
                                        )],
                                    ),
                    ],
                        className="twelve columns"
                    ),
                ], className="row"
                ),
            ],
                className='eight columns'
            ),
        ],
            className='row'
        ),
        html.H5('Window Size',
                style={'fontSize': 20},
                id='header-slider-window-size'),
        dcc.Slider(10, 300, 5,
                   value=30,
                   id='window-size'
                   ),
        html.H5('Step Size',
                style={'fontSize': 20},
                id='header-slider-step-size'),
        dcc.Slider(1, 100, 1,
                   value=50,
                   id='step-size'
                   ),
    ],
        style=styles['div'],
        id='change-scatter-container'
    ),
    html.Div(children=[
        dcc.Loading(id="loading-change-angle",
                    type="default",
                    children=[
                        dcc.Graph(
                            id='change-angle-plot',
                            figure={}
                        )],
                    ),
        html.H5('Minimal Radius',
                style={'fontSize': 20},
                id='header-slider-radius'),
        dcc.Slider(0, 100, 1,
                   value=5,
                   id='min-radius'
                   ),
        html.H5('Smoothing Window',
                style={'fontSize': 20},
                id='header-slider-smoothing-size'),
        dcc.Input(id='smoothing-size', type="number", placeholder="20", value=200, min=1, debounce=True)
        ],
        style=styles['div'],
        id='change-angle-container'
    ),
    html.Div([
        dcc.Markdown("""
                    **Selection Data**

                    Choose the lasso or rectangle tool in the graph's menu
                    bar and then select points in the graph.
                    
                """),
        html.Pre(id='selected-data', style=styles['pre'])],
        style=styles['div'],
        hidden=False,
        id='selected-data-container'
    )
],
)


@callback(
    Output('change-scatter', 'figure'),
    Input("loading-1", "children"),
    Input("window-size", "value"),
    Input("step-size", "value"),
    Input('signal-graph', 'relayoutData')
)
def compute_and_plot(value, window, step, relayoutData):

    # time the computations
    start = time.perf_counter()

    # compute the change score using a cached function
    flow_t, flow, flow_change, pressure_t, pressure, pressure_change = compute_change_score(window, step)

    # restrict the timing in accordance with the zoom
    flow_t, flow, flow_change, pressure_t, pressure, pressure_change = restrict_time(flow_t, flow, flow_change,
                                                                                     pressure_t, pressure,
                                                                                     pressure_change, relayoutData)

    # plot the scatter plot
    fig = px.scatter(pd.DataFrame({'pressure change': pressure_change,
                                   'flow change': flow_change,
                                   'time_array': flow_t,
                                   'Time': ['{0:02.0f} h:{1:02.0f} min'.format(*divmod(timed * 60, 60)) for timed in flow_t],
                                   'Diff.': ['dFlow > dPress' if dFlow > dPress else 'dFlow < dPress' for (dFlow, dPress) in zip(flow_change, pressure_change)],
                                   }),
                     x='pressure change',
                     y='flow change',
                     hover_data=['pressure change', 'flow change', 'time_array', 'Time', 'Diff.'],
                     title=f"Change Scatter for HS2 flow against pressure\nwith window size {window} and step "
                           f"size {step}")
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], name='Main Diag.', line=dict(color='black', width=4, dash='dash'))
    )
    fig['data'][1]['showlegend'] = False
    print(f"Computation for window size {window} and step size {step} took: {time.perf_counter() - start} s.")
    return fig


@callback(
    Output('flow-change-plot', 'figure'),
    Output('pressure-change-plot', 'figure'),
    Input("loading-flow-change", "children"),
    Input("loading-pressure-change", "children"),
    Input("loading-change-angle", "children"),
    Input("window-size", "value"),
    Input("step-size", "value"),
    Input('signal-graph', 'relayoutData')
)
def slider_change_parameters(value1, value2, value3, window, step, relayoutData):

    # compute the change score using a cached function
    flow_t, flow, flow_change, pressure_t, pressure, pressure_change = compute_change_score(window, step)

    # restrict the timing in accordance with the zoom
    flow_t, flow, flow_change, pressure_t, pressure, pressure_change = restrict_time(flow_t, flow, flow_change,
                                                                                     pressure_t, pressure,
                                                                                     pressure_change, relayoutData)

    # Create a figure with secondary y-axis
    flow_fig = make_subplots(specs=[[{"secondary_y": True}]])
    pressure_fig = make_subplots(specs=[[{"secondary_y": True}]])

    # make the flow change plot
    flow_fig.add_trace(
        go.Scatter(x=flow_t, y=flow, name="Flow Signal", line=dict(color="orange")),
        secondary_y=False,
    )
    flow_fig.add_trace(
        go.Scatter(x=flow_t, y=flow_change, name="Change Score"),
        secondary_y=True,
    )

    # make the pressure change plot
    pressure_fig.add_trace(
        go.Scatter(x=pressure_t, y=pressure, name="Pressure Signal", line=dict(color="green")),
        secondary_y=False,
    )
    pressure_fig.add_trace(
        go.Scatter(x=pressure_t, y=pressure_change, name="Change Score"),
        secondary_y=True,
    )
    return flow_fig, pressure_fig


@callback(
    Output('selected-data', 'children', allow_duplicate=True),
    Output('signal-graph', 'figure', allow_duplicate=True),
    Output('flow-change-plot', 'figure', allow_duplicate=True),
    Output('pressure-change-plot', 'figure', allow_duplicate=True),
    Output('change-angle-plot', 'figure', allow_duplicate=True),
    Input('change-scatter', 'clickData'),
    Input('change-scatter', 'selectedData'),
    Input('signal-graph', 'figure'),
    Input('flow-change-plot', 'figure'),
    Input('pressure-change-plot', 'figure'),
    Input('change-angle-plot', 'figure'),
    prevent_initial_call=True)
def interact_scatter_plot(click_data, selected_data, figure_scatter, figure_change_flow, figure_change_pressure,
                          figure_change_angle):

    # check that we do not redo this when only the figures get updated
    if click_data is None and selected_data is None:
        return no_update, no_update, no_update, no_update, no_update
    print("CLICKED", click_data, selected_data)
    # check whether we clicked a point
    if not click_data or not click_data['points']:
        click_data = None
    else:
        click_data = click_data['points'][0]['customdata'][0]

    # check whether we selected multiple points
    if not selected_data or not selected_data['points']:
        selected_data = None
    else:
        selected_data = [point['customdata'][0] for point in selected_data['points']]

    # update all the figures
    figure_scatter = update_marks(figure_scatter, click_data, selected_data)
    figure_change_flow = update_marks(figure_change_flow, click_data, selected_data,
                                      first_name='Flow Signal', second_name='Change Score')
    figure_change_pressure = update_marks(figure_change_pressure, click_data, selected_data,
                                          first_name='Pressure Signal', second_name='Change Score')
    figure_change_angle = update_marks(figure_change_angle, click_data, selected_data,
                                       first_name='Change Angles', second_name='Change Angles')
    return str(click_data), figure_scatter, figure_change_flow, figure_change_pressure, figure_change_angle


@callback(
    Output('change-scatter', 'clickData', allow_duplicate=True),
    Output('change-scatter', 'selectedData', allow_duplicate=True),
    Input('change-angle-plot', 'clickData'),
    Input('change-angle-plot', 'selectedData'),
    prevent_initial_call=True)
def click_change_angle(click_data, selected_data):

    # check whether we clicked a point and construct an update as the chane scatter update function expects it
    if not click_data or not click_data['points']:
        click_data = no_update
    else:
        click_data = {"points": [{'customdata': [click_data['points'][0]['x']]}]}

    # check whether we selected multiple points and construct an update as the change scatter update function expects it
    if not selected_data or not selected_data['points']:
        selected_data = no_update
    else:
        click_data = {"points": [{'customdata': [ele['x']]} for ele in click_data['points']]}
    return click_data, selected_data


@callback(
    Output('change-scatter', 'clickData', allow_duplicate=True),
    Output('change-scatter', 'selectedData', allow_duplicate=True),
    Input('flow-change-plot', 'clickData'),
    Input('flow-change-plot', 'selectedData'),
    prevent_initial_call=True)
def click_flow_change(click_data, selected_data):

    # check whether we clicked a point and construct an update as the chane scatter update function expects it
    if not click_data or not click_data['points']:
        click_data = no_update
    else:
        click_data = {"points": [{'customdata': [click_data['points'][0]['x']]}]}

    # check whether we selected multiple points and construct an update as the change scatter update function expects it
    if not selected_data or not selected_data['points']:
        selected_data = no_update
    else:
        click_data = {"points": [{'customdata': [ele['x']]} for ele in click_data['points']]}
    return click_data, selected_data


@callback(
    Output('change-scatter', 'clickData', allow_duplicate=True),
    Output('change-scatter', 'selectedData', allow_duplicate=True),
    Input('pressure-change-plot', 'clickData'),
    Input('pressure-change-plot', 'selectedData'),
    prevent_initial_call=True)
def click_pressure_change(click_data, selected_data):

    # check whether we clicked a point and construct an update as the chane scatter update function expects it
    if not click_data or not click_data['points']:
        click_data = no_update
    else:
        click_data = {"points": [{'customdata': [click_data['points'][0]['x']]}]}

    # check whether we selected multiple points and construct an update as the change scatter update function expects it
    if not selected_data or not selected_data['points']:
        selected_data = no_update
    else:
        click_data = {"points": [{'customdata': [ele['x']]} for ele in click_data['points']]}
    return click_data, selected_data


@callback(
    Output('change-scatter', 'clickData', allow_duplicate=True),
    Output('change-scatter', 'selectedData', allow_duplicate=True),
    Input('signal-graph', 'clickData'),
    Input('signal-graph', 'selectedData'),
    prevent_initial_call=True)
def click_signal(click_data, selected_data):
    print(click_data, 'asdasd')
    # check whether we clicked a point and construct an update as the chane scatter update function expects it
    if not click_data or not click_data['points']:
        click_data = no_update
    else:
        click_data = {"points": [{'customdata': [click_data['points'][0]['x']]}]}

    # check whether we selected multiple points and construct an update as the change scatter update function expects it
    if not selected_data or not selected_data['points']:
        selected_data = no_update
    else:
        click_data = {"points": [{'customdata': [ele['x']]} for ele in click_data['points']]}
    return click_data, selected_data


@callback(
    Output('change-angle-plot', 'figure'),
    Input("loading-change-angle", "children"),
    Input("window-size", "value"),
    Input("step-size", "value"),
    Input("min-radius", "value"),
    Input("smoothing-size", "value"),
    Input('signal-graph', 'relayoutData'),
)
def compute_change_angle(value, window, step, radius, smoothing_size, relayoutData):

    # compute the change score using a cached function
    flow_t, flow, flow_change, pressure_t, pressure, pressure_change = compute_change_score(window, step)

    # restrict the timing in accordance with the zoom
    flow_t, flow, flow_change, pressure_t, pressure, pressure_change = restrict_time(flow_t, flow, flow_change,
                                                                                     pressure_t, pressure,
                                                                                     pressure_change, relayoutData)

    # TODO: Make better smoothing? -> We would need that for coupling degree
    # compute a sliding mean
    sliding_flow_change = np.convolve(flow_change, np.ones(smoothing_size) / smoothing_size, mode='valid')
    sliding_pressure_change = np.convolve(pressure_change, np.ones(smoothing_size) / smoothing_size, mode='valid')
    sliding_pressure_change[sliding_pressure_change <= np.finfo("float").eps * 10] = np.finfo("float").eps * 10

    # build a mask for points where both values are very small
    mask = sliding_pressure_change * sliding_pressure_change + sliding_flow_change * sliding_flow_change
    mask = mask <= (radius/100) ** 2

    coupling_angle = np.rad2deg(np.arctan(sliding_flow_change / sliding_pressure_change)) - 45
    coupling_angle[mask] = np.nan
    padding_left = np.empty(math.ceil((smoothing_size-1)/2))
    padding_left[:] = np.nan
    padding_right = np.empty(math.floor((smoothing_size-1)/2))
    padding_right[:] = np.nan
    coupling_angle = np.concatenate((padding_left, coupling_angle, padding_right))
    assert len(coupling_angle) == len(pressure_change)

    # make the change angle
    angle_fig = make_subplots(specs=[[{"secondary_y": True}]])
    angle_fig = angle_fig.add_trace(go.Scatter(x=flow_t, y=coupling_angle, name="Change Angles", connectgaps=False))
    angle_fig['data'][0]['showlegend'] = True
    return angle_fig

@functools.cache
def compute_change_score(window_size: int, step_size: int) \
        -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):

    # get the signals
    flow_t, flow, pressure_t, pressure = read_data()

    # make the scorer and compute the score
    # scorer = SST(window_length=window_size, scoring_step=step_size, method="rsvd")
    scorer = ESST(window_length=window_size, scoring_step=step_size)
    flow_change = scorer.transform(flow)
    pressure_change = scorer.transform(pressure)

    # normalize the scores
    flow_iqr = np.max(flow_change) - np.min(flow_change)
    pressure_iqr = np.max(pressure_change) - np.min(pressure_change)
    if flow_iqr:
        flow_change = (flow_change-np.min(flow_change))/flow_iqr
    if pressure_iqr:
        pressure_change = (pressure_change-np.min(pressure_change))/pressure_iqr

    return flow_t, flow, flow_change, pressure_t, pressure, pressure_change


def restrict_time(flow_t, flow, flow_change, pressure_t, pressure, pressure_change, relayoutData):

    # check whether we have and updated x-axis-range
    if relayoutData is not None and 'xaxis.range[0]' in relayoutData and 'xaxis.range[1]' in relayoutData:
        # get the time restrictions
        x_min = relayoutData['xaxis.range[0]']
        x_max = relayoutData['xaxis.range[1]']

        # find the indices
        idx = np.where(np.logical_and(flow_t > x_min, flow_t < x_max))[0]

        # restrict the signals
        flow_t = flow_t[idx]
        flow = flow[idx]
        flow_change = flow_change[idx]

        # find the indices
        idx = np.where(np.logical_and(pressure_t > x_min, pressure_t < x_max))[0]
        pressure_t = pressure_t[idx]
        pressure = pressure[idx]
        pressure_change = pressure_change[idx]
    return flow_t, flow, flow_change, pressure_t, pressure, pressure_change


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8080)
