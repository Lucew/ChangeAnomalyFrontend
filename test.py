from dash import Dash, dcc, html, Input, Output, callback
import time

app = Dash(__name__)
app.scripts.config.serve_locally = True
app.css.config.serve_locally = True

app.layout = html.Div(
    children=[
        html.H3("Edit text input to see loading state"),
        dcc.Input(id="loading-input-1", value='Input triggers local spinner'),
        dcc.Loading(
            id="loading-1",
            type="default",
            children=html.Div(id="loading-output-1")
        ),
        html.Div(
            [
                dcc.Input(id="loading-input-2", value='Input triggers nested spinner'),
                dcc.Loading(
                    id="loading-2",
                    children=[html.Div([html.Div(id="loading-output-2")])],
                    type="circle",
                )
            ]
        ),
    ],
)


@callback(Output("loading-output-1", "children"), Input("loading-input-1", "value"))
def input_triggers_spinner(value):
    time.sleep(1)
    return value


@callback(Output("loading-output-2", "children"), Input("loading-input-2", "value"))
def input_triggers_nested(value):
    time.sleep(1)
    return value


if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')